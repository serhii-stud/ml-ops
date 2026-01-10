import argparse
import os
import subprocess
import sys
import time
import uuid
import tarfile
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List


def install_requirements(req_path: str) -> None:
    if not os.path.exists(req_path):
        raise FileNotFoundError(f"requirements.txt not found: {req_path}")
    print(f"[INFO] Installing dependencies from: {req_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("[INFO] Dependencies installed.")


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


# NEW: Make a string safe for SageMaker resource names.
# SageMaker names should be alphanumeric and hyphen, avoid long strings.
def _sm_safe_name(s: str, max_len: int = 63) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9-]+", "-", s)     # replace unsupported chars with hyphen
    s = re.sub(r"-{2,}", "-", s).strip("-")   # collapse multiple hyphens
    if not s:
        s = "mlflow-run"
    return s[:max_len]


# NEW: Extract "human" MLflow run name (mlflow.runName) by run_id; fallback to run_id.
def _get_mlflow_run_name(client, run_id: str) -> str:
    run = client.get_run(run_id)
    tags = run.data.tags or {}
    return tags.get("mlflow.runName") or run.info.run_id


def resolve_champion_uri(
    client,
    registered_model_name: str,
    aliases: List[str],
    stages: List[str],
    champion_tag_key: str,
) -> Tuple[str, str, object]:
    for alias in aliases:
        try:
            mv = client.get_model_version_by_alias(registered_model_name, alias)
            return f"models:/{registered_model_name}@{alias}", f"alias:{alias} -> v{mv.version}", mv
        except Exception:
            pass

    for stage in stages:
        try:
            mvs = client.get_latest_versions(registered_model_name, stages=[stage])
            if mvs:
                mv = mvs[0]
                return f"models:/{registered_model_name}/{stage}", f"stage:{stage} -> v{mv.version}", mv
        except Exception:
            pass

    versions = client.search_model_versions(f"name='{registered_model_name}'")
    tagged = [mv for mv in versions if _truthy((mv.tags or {}).get(champion_tag_key))]
    if tagged:
        tagged.sort(key=lambda x: x.creation_timestamp or 0, reverse=True)
        mv = tagged[0]
        return f"models:/{registered_model_name}/{mv.version}", f"tag:{champion_tag_key}=true -> v{mv.version}", mv

    raise RuntimeError("Champion model not found")


def deploy_sagemaker_endpoint(
    region: str,
    endpoint_name: str,
    model_uri: str,
    endpoint_execution_role_arn: str,
    image_url: str,
    instance_type: str,
    instance_count: int,
    bucket_name: str,
    timeout_seconds: int,
    registered_model_name: str,
    mlflow_run_name: str,
    mlflow_model_version: str,
) -> None:
    import boto3
    import mlflow

    sm = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:8]

    print(f"[INFO] Downloading MLflow artifacts for {model_uri}")
    local_dir = Path(mlflow.artifacts.download_artifacts(model_uri))

    tar_path = Path("/tmp") / f"model_{ts}_{uid}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in local_dir.rglob("*"):
            tar.add(p, arcname=p.relative_to(local_dir))

    s3_key = f"sagemaker-deploy/{endpoint_name}/{ts}_{uid}/model.tar.gz"
    print(f"[INFO] Uploading model to s3://{bucket_name}/{s3_key}")
    s3.upload_file(str(tar_path), bucket_name, s3_key)

    # NEW: Use MLflow run name in SageMaker ModelName (sanitized) + keep uniqueness suffix.
    safe_run = _sm_safe_name(mlflow_run_name, max_len=48)
    safe_ver = _sm_safe_name(f"v{mlflow_model_version}", max_len=10)
    model_name = _sm_safe_name(f"{safe_run}-{safe_ver}-{uid}", max_len=63)

    model_data_url = f"s3://{bucket_name}/{s3_key}"

    tags = [
        {"Key": "mlflow_model", "Value": registered_model_name},
        {"Key": "mlflow_model_uri", "Value": model_uri},
        {"Key": "mlflow_run_name", "Value": mlflow_run_name},          # NEW: store run name as tag too
        {"Key": "mlflow_model_version", "Value": str(mlflow_model_version)},  # NEW
        {"Key": "deployed_at", "Value": ts},
    ]

    print(f"[INFO] Creating SageMaker model {model_name}")
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=endpoint_execution_role_arn,
        PrimaryContainer={
            "Image": image_url,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "MLFLOW_DISABLE_ENV_CREATION": "true",
                "MLFLOW_ENV_MANAGER": "local",
                "ENDPOINT_NAME": endpoint_name,
                "MODEL_VERSION": str(mlflow_model_version),   # или mv.run_id, или и то и то
                "MLFLOW_RUN_ID": mv.run_id,                   # если хочешь
                "METRICS_NAMESPACE": "MLOps/BankingEndpoint",
                "METRICS_SAMPLE_RATE": "0.2"
            },
        },
        Tags=tags,
    )

    # NEW: Also include run name in EndpointConfigName for traceability.
    endpoint_config_name = _sm_safe_name(f"{endpoint_name}-cfg-{safe_run}-{safe_ver}-{uid}", max_len=63)

    print(f"[INFO] Creating EndpointConfig {endpoint_config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": instance_count,
                "InitialVariantWeight": 1.0,
            }
        ],
        Tags=tags
    )

    try:
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"[INFO] Updating existing endpoint {endpoint_name}")
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )
    except sm.exceptions.ClientError:
        print(f"[INFO] Creating new endpoint {endpoint_name}")
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            Tags=tags
        )

    print("[INFO] Waiting for endpoint to become InService...")
    deadline = time.time() + timeout_seconds
    while True:
        status = sm.describe_endpoint(EndpointName=endpoint_name)["EndpointStatus"]
        print(f"[INFO] Endpoint status: {status}")
        if status == "InService":
            break
        if status in {"Failed", "OutOfService"}:
            raise RuntimeError(f"Endpoint failed: {status}")
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for endpoint")
        time.sleep(15)

    print("[SUCCESS] Deployment completed")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mlflow_tracking_uri", required=True)
    p.add_argument("--registered_model_name", required=True)
    p.add_argument("--alias", default="champion")
    p.add_argument("--alias_fallback", default="chempion")
    p.add_argument("--stage", default="Production")
    p.add_argument("--stage_fallback", default="Prod")
    p.add_argument("--champion_tag_key", default="champion")
    p.add_argument("--region", required=True)
    p.add_argument("--endpoint_name", required=True)
    p.add_argument("--endpoint_execution_role_arn", required=True)
    p.add_argument("--image_url", required=True)
    p.add_argument("--instance_type", default="ml.m5.large")
    p.add_argument("--instance_count", type=int, default=1)
    p.add_argument("--bucket_name", required=True)
    p.add_argument("--timeout_seconds", type=int, default=1200)
    p.add_argument("--requirements", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    install_requirements(args.requirements)

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    client = MlflowClient()

    model_uri, reason, mv = resolve_champion_uri(
        client,
        args.registered_model_name,
        [args.alias, args.alias_fallback],
        [args.stage, args.stage_fallback],
        args.champion_tag_key,
    )

    print(f"[INFO] Champion resolved: {model_uri} ({reason})")

    # NEW: Resolve MLflow run name (e.g., "banking-train-...") from the champion model version.
    mlflow_run_name = _get_mlflow_run_name(client, mv.run_id)
    mlflow_model_version = str(mv.version)

    print(f"[INFO] MLflow run_name resolved: {mlflow_run_name}")
    print(f"[INFO] MLflow model version: {mlflow_model_version}")

    deploy_sagemaker_endpoint(
        region=args.region,
        endpoint_name=args.endpoint_name,
        model_uri=model_uri,
        endpoint_execution_role_arn=args.endpoint_execution_role_arn,
        image_url=args.image_url,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        bucket_name=args.bucket_name,
        timeout_seconds=args.timeout_seconds,
        registered_model_name=args.registered_model_name,
        mlflow_run_name=mlflow_run_name,
        mlflow_model_version=mlflow_model_version,
    )


if __name__ == "__main__":
    main()
