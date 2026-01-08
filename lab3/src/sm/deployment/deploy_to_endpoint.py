import argparse
import os
import subprocess
import sys
from typing import Optional, Tuple, List


def install_requirements(req_path: str) -> None:
    """Install dependencies inside the Processing container."""
    if not os.path.exists(req_path):
        raise FileNotFoundError(f"requirements.txt not found: {req_path}")
    print(f"[INFO] Installing dependencies from: {req_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("[INFO] Dependencies installed.")


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def resolve_champion_uri(
    client,
    registered_model_name: str,
    aliases: List[str],
    stages: List[str],
    champion_tag_key: str,
) -> Tuple[str, str]:
    """
    Resolve the champion model in this priority order:
    1) alias (e.g., champion / chempion)
    2) stage (e.g., Production / Prod)
    3) tag champion=true on a model version (latest by creation_timestamp)
    """
    # 1) Aliases
    for alias in aliases:
        if not alias:
            continue
        try:
            mv = client.get_model_version_by_alias(registered_model_name, alias)
            return f"models:/{registered_model_name}@{alias}", f"alias:{alias} -> v{mv.version}"
        except Exception:
            pass

    # 2) Stages (latest in stage)
    for stage in stages:
        if not stage:
            continue
        try:
            mvs = client.get_latest_versions(registered_model_name, stages=[stage])
            if mvs:
                return f"models:/{registered_model_name}/{stage}", f"stage:{stage} -> v{mvs[0].version}"
        except Exception:
            pass

    # 3) Tag on model versions
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    tagged = []
    for mv in versions:
        tag_val = (mv.tags or {}).get(champion_tag_key)
        if _truthy(tag_val):
            tagged.append(mv)

    if tagged:
        tagged.sort(key=lambda x: x.creation_timestamp or 0, reverse=True)
        best = tagged[0]
        return f"models:/{registered_model_name}/{best.version}", f"tag:{champion_tag_key}=true -> v{best.version}"

    raise RuntimeError(
        f"Champion not found for '{registered_model_name}'. "
        f"Tried aliases={aliases}, stages={stages}, tag_key={champion_tag_key}."
    )


def deploy_sagemaker_endpoint(
    region: str,
    endpoint_name: str,
    model_uri: str,
    endpoint_execution_role_arn: str,
    image_url: str,
    instance_type: str,
    instance_count: int,
    bucket_name: Optional[str],
    timeout_seconds: int,
) -> None:
    from mlflow.deployments import get_deploy_client

    deploy_client = get_deploy_client(f"sagemaker:/{region}")

    config = {
        "execution_role_arn": endpoint_execution_role_arn,
        "image_url": image_url,
        "region_name": region,
        "instance_type": instance_type,
        "instance_count": instance_count,
        "synchronous": True,
        "timeout_seconds": timeout_seconds,
        "archive": True,
        "env": {
            "MLFLOW_ENV_MANAGER": "local",
            "MLFLOW_DISABLE_ENV_CREATION": "true",
        },
    }
    if bucket_name:
        config["bucket_name"] = bucket_name

    print(f"[INFO] Deploying '{model_uri}' to endpoint '{endpoint_name}' in region '{region}'")
    deploy_client.create_deployment(
        name=endpoint_name,
        model_uri=model_uri,
        config=config,
    )
    print("[SUCCESS] Deployment completed.")


def parse_args():
    p = argparse.ArgumentParser()

    # MLflow
    p.add_argument("--mlflow_tracking_uri", required=True)
    p.add_argument("--registered_model_name", required=True)

    # Champion resolution
    p.add_argument("--alias", default="champion")
    p.add_argument("--alias_fallback", default="chempion")
    p.add_argument("--stage", default="Production")
    p.add_argument("--stage_fallback", default="Prod")
    p.add_argument("--champion_tag_key", default="champion")

    # SageMaker
    p.add_argument("--region", required=True)
    p.add_argument("--endpoint_name", required=True)
    p.add_argument("--endpoint_execution_role_arn", required=True)
    p.add_argument("--image_url", required=True)
    p.add_argument("--instance_type", default="ml.m5.large")
    p.add_argument("--instance_count", type=int, default=1)
    p.add_argument("--bucket_name", default="")
    p.add_argument("--timeout_seconds", type=int, default=1200)

    # Requirements file mounted by ProcessingInput
    p.add_argument("--requirements", required=True)

    return p.parse_args()


def main():
    args = parse_args()

    # Install dependencies first (mlflow may need extras)
    install_requirements(args.requirements)

    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.deployments import get_deploy_client
    
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    client = MlflowClient()

    aliases = [args.alias, args.alias_fallback]
    stages = [args.stage, args.stage_fallback]

    model_uri, reason = resolve_champion_uri(
        client=client,
        registered_model_name=args.registered_model_name,
        aliases=aliases,
        stages=stages,
        champion_tag_key=args.champion_tag_key,
    )
    print(f"[INFO] Champion resolved: {model_uri} ({reason})")

    deploy_sagemaker_endpoint(
        region=args.region,
        endpoint_name=args.endpoint_name,
        model_uri=model_uri,
        endpoint_execution_role_arn=args.endpoint_execution_role_arn,
        image_url=args.image_url,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        bucket_name=(args.bucket_name or None),
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
