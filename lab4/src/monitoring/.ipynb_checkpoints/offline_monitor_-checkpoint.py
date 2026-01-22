import argparse
import json
import os
import re
import subprocess
import sys
import csv
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict

# -----------------------------
# Bootstrap deps (pin compatible versions)
# -----------------------------

def install_requirements(req_path: str) -> None:
    if not req_path:
        return
    if not os.path.exists(req_path):
        raise FileNotFoundError(f"requirements.txt not found: {req_path}")
    print(f"[INFO] Installing dependencies from: {req_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("[INFO] Dependencies installed.")


import boto3
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri}")
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def sanitize_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:200] if s else "file"


# -----------------------------
# Data Capture reading
# -----------------------------
def list_s3_objects_modified_after(
    s3,
    bucket: str,
    prefix: str,
    cutoff: datetime,
    max_keys: int = 4000,
) -> List[Tuple[str, datetime]]:
    out: List[Tuple[str, datetime]] = []
    token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)

        for obj in resp.get("Contents", []):
            lm = obj["LastModified"]
            if lm >= cutoff:
                out.append((obj["Key"], lm))
                if len(out) >= max_keys:
                    out.sort(key=lambda x: x[1])
                    return out

        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    out.sort(key=lambda x: x[1])
    return out


def download_s3_objects(s3, bucket: str, keys: List[str], local_dir: str) -> List[str]:
    os.makedirs(local_dir, exist_ok=True)
    local_paths = []
    for key in keys:
        name = sanitize_filename(key.split("/")[-1])
        path = os.path.join(local_dir, name)
        s3.download_file(bucket, key, path)
        local_paths.append(path)
    return local_paths


def parse_data_capture_jsonl(files: List[str]) -> pd.DataFrame:
    """
    Your endpoint call:
      input : {"inputs": ["<text>"]}
      output: {"predictions": ["<label>"]}
    """
    rows: List[Dict] = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = safe_json_loads(line)
                if not isinstance(rec, dict):
                    continue

                event_time = rec.get("eventTime")
                inp = (rec.get("endpointInput") or {}).get("data", "")
                out = (rec.get("endpointOutput") or {}).get("data", "")

                # Input
                text = ""
                inp_obj = safe_json_loads(inp) if isinstance(inp, str) else None
                if isinstance(inp_obj, dict):
                    inputs = inp_obj.get("inputs")
                    if isinstance(inputs, list) and inputs:
                        text = str(inputs[0])
                    else:
                        text = json.dumps(inp_obj, ensure_ascii=False)
                else:
                    text = inp if isinstance(inp, str) else str(inp)

                # Output
                pred_label = "unknown"
                out_obj = safe_json_loads(out) if isinstance(out, str) else None
                if isinstance(out_obj, dict):
                    preds = out_obj.get("predictions")
                    if isinstance(preds, list) and preds:
                        pred_label = str(preds[0])
                elif isinstance(out, str) and out.strip():
                    pred_label = out.strip()

                rows.append(
                    {
                        "event_time": event_time,
                        "text": text,
                        "text_len": len(text) if text else 0,
                        "pred_label": pred_label,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["event_time", "text", "text_len", "pred_label"])
    return df


# -----------------------------
# Reference dataset
# -----------------------------
def load_reference_train(s3, s3_uri: str) -> pd.DataFrame:
    b, k = parse_s3_uri(s3_uri)
    local_path = "/opt/ml/processing/reference/train.csv"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(b, k, local_path)
    return pd.read_csv(local_path)


def build_reference_features(ref: pd.DataFrame) -> pd.DataFrame:
    if "text" in ref.columns:
        text_len = ref["text"].astype(str).str.len()
    elif "text_len" in ref.columns:
        text_len = ref["text_len"].astype(int)
    else:
        raise RuntimeError("Reference train.parquet must contain 'text' or 'text_len' column")
    return pd.DataFrame({"text_len": text_len})


def build_current_features(cur: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"text_len": cur["text_len"].astype(int)})


# -----------------------------
# Offline ML metrics (no confidence)
# -----------------------------
def compute_offline_metrics(cur: pd.DataFrame, unknown_label: str = "unknown") -> dict:
    n = int(len(cur))
    if n == 0:
        return {"n": 0}

    empty_ratio = float((cur["text_len"] == 0).mean())
    short_ratio = float((cur["text_len"] < 5).mean())
    len_mean = float(cur["text_len"].mean())

    vc = cur["pred_label"].fillna(unknown_label).astype(str).value_counts()
    shares = {cls: float(cnt / n) for cls, cnt in vc.items()}
    unknown_ratio = float(shares.get(unknown_label, 0.0))

    return {
        "n": n,
        "empty_text_ratio": empty_ratio,
        "short_text_ratio": short_ratio,
        "text_length_mean": len_mean,
        "unknown_ratio": unknown_ratio,
        "pred_class_share": shares,
    }


def put_cloudwatch_metrics(cw, namespace: str, dims: List[dict], metrics: dict):
    now = utcnow()
    metric_data = []

    def add(name: str, value: Optional[float], unit: str = "None", extra_dims: Optional[List[dict]] = None):
        if value is None:
            return
        metric_data.append(
            {
                "MetricName": name,
                "Dimensions": dims + (extra_dims or []),
                "Timestamp": now,
                "Value": float(value),
                "Unit": unit,
            }
        )

    add("empty_text_ratio", metrics.get("empty_text_ratio"))
    add("short_text_ratio", metrics.get("short_text_ratio"))
    add("text_length_mean", metrics.get("text_length_mean"))
    add("unknown_ratio", metrics.get("unknown_ratio"))

    for cls, share in (metrics.get("pred_class_share") or {}).items():
        add("pred_class_share", share, extra_dims=[{"Name": "Class", "Value": str(cls)}])

    for i in range(0, len(metric_data), 20):
        cw.put_metric_data(Namespace=namespace, MetricData=metric_data[i : i + 20])


# -----------------------------
# Evidently drift
# -----------------------------
def run_evidently(ref_feat: pd.DataFrame, cur_feat: pd.DataFrame, out_dir: str) -> dict:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.test_suite import TestSuite
    from evidently.test_preset import DataDriftTestPreset
    
    os.makedirs(out_dir, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_feat, current_data=cur_feat)
    report_html = os.path.join(out_dir, "drift_report.html")
    report_json = os.path.join(out_dir, "drift_report.json")
    report.save_html(report_html)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report.as_dict(), f, ensure_ascii=False)

    suite = TestSuite(tests=[DataDriftTestPreset()])
    suite.run(reference_data=ref_feat, current_data=cur_feat)
    suite_dict = suite.as_dict()
    with open(os.path.join(out_dir, "drift_tests.json"), "w", encoding="utf-8") as f:
        json.dump(suite_dict, f, ensure_ascii=False)

    failed = int(suite_dict.get("summary", {}).get("failed", 0))
    total = int(suite_dict.get("summary", {}).get("total", 0))
    drift_share = float(failed / max(total, 1))

    return {
        "failed_tests_count": failed,
        "total_tests_count": total,
        "drift_share": drift_share,
        "report_html": report_html,
        "report_json": report_json,
    }


def write_current_csv(cur: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["event_time", "text", "text_len", "pred_label"]

    # На случай если колонок нет (пустой df)
    for c in cols:
        if c not in cur.columns:
            cur[c] = "" if c in ("event_time", "text", "pred_label") else 0

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in cur[cols].to_dict(orient="records"):
            w.writerow(row)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_s3_prefix", required=True)
    ap.add_argument("--reference_train_s3_uri", required=True)
    ap.add_argument("--output_s3_prefix", required=True)
    ap.add_argument("--lookback_hours", type=int, default=24)
    ap.add_argument("--endpoint_name", required=True)
    ap.add_argument("--effective_version", required=True)
    ap.add_argument("--namespace", default="MLOps/BankingEndpoint")
    ap.add_argument("--unknown_label", default="unknown")
    ap.add_argument("--requirements", type=str, default="")
    ap.add_argument("--aws_region", type=str, default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "")
    args = ap.parse_args()

    install_requirements(args.requirements)

    region = args.aws_region.strip()
    if not region:
        raise RuntimeError(
            "AWS region is not set. Pass --aws_region or set AWS_REGION/AWS_DEFAULT_REGION."
        )
    
    s3 = boto3.client("s3", region_name=region)
    cw = boto3.client("cloudwatch", region_name=region)
    print("[INFO] Using AWS region:", region)

    # Reference
    ref = load_reference_train(s3, args.reference_train_s3_uri)
    ref_feat = build_reference_features(ref)

    # Current window
    cap_bucket, cap_prefix = parse_s3_uri(args.capture_s3_prefix)
    cutoff = utcnow() - timedelta(hours=args.lookback_hours)

    objs = list_s3_objects_modified_after(s3, cap_bucket, cap_prefix, cutoff=cutoff)
    keys = [k for (k, _lm) in objs]
    print(f"[INFO] Found {len(keys)} capture objects in last {args.lookback_hours}h under {args.capture_s3_prefix}")

    local_cap_dir = "/opt/ml/processing/current/capture"
    local_files = download_s3_objects(s3, cap_bucket, keys, local_cap_dir) if keys else []

    cur = parse_data_capture_jsonl(local_files)

    out_dir = "/opt/ml/processing/output"
    os.makedirs(out_dir, exist_ok=True)

    cur_csv = os.path.join(out_dir, "current_inference.csv")
    #cur.to_csv(cur_csv, index=False)
    write_current_csv(cur, cur_csv)

    # Offline metrics
    offline = compute_offline_metrics(cur, unknown_label=args.unknown_label)

    # Evidently
    drift_dir = os.path.join(out_dir, "evidently")
    cur_feat = build_current_features(cur)
    drift_summary = run_evidently(ref_feat, cur_feat, drift_dir)

    # CloudWatch
    dims = [
        {"Name": "EndpointName", "Value": args.endpoint_name},
        {"Name": "EffectiveVersion", "Value": args.effective_version},
    ]
    put_cloudwatch_metrics(cw, args.namespace, dims, offline)
    cw.put_metric_data(
        Namespace=args.namespace,
        MetricData=[
            {
                "MetricName": "drift_share",
                "Dimensions": dims,
                "Timestamp": utcnow(),
                "Value": float(drift_summary["drift_share"]),
                "Unit": "None",
            },
            {
                "MetricName": "failed_tests_count",
                "Dimensions": dims,
                "Timestamp": utcnow(),
                "Value": float(drift_summary["failed_tests_count"]),
                "Unit": "Count",
            },
        ],
    )

    # Upload artifacts
    out_bucket, out_prefix = parse_s3_uri(args.output_s3_prefix)
    stamp = utcnow().strftime("%Y%m%d-%H%M%S")
    base = f"{out_prefix.rstrip('/')}/offline_monitor/{args.endpoint_name}/{args.effective_version}/{stamp}"

    metrics_path = os.path.join(out_dir, "metrics.json")
    payload = {
        **offline,
        **{k: drift_summary[k] for k in ("failed_tests_count", "total_tests_count", "drift_share")},
        "lookback_hours": args.lookback_hours,
        "cutoff_utc": cutoff.isoformat(),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    uploads = [
        (metrics_path, "metrics.json"),
        (cur_csv, "current_inference.csv"),
        (drift_summary["report_html"], "drift_report.html"),
        (drift_summary["report_json"], "drift_report.json"),
        (os.path.join(drift_dir, "drift_tests.json"), "drift_tests.json"),
    ]

    for local_path, name in uploads:
        if os.path.exists(local_path):
            key = f"{base}/{name}"
            s3.upload_file(local_path, out_bucket, key)
            print(f"[INFO] Uploaded: s3://{out_bucket}/{key}")
        else:
            print(f"[WARN] Missing artifact: {local_path}")

    print("[SUCCESS] Offline monitoring completed.")
    print("Artifacts prefix:", f"s3://{out_bucket}/{base}/")


if __name__ == "__main__":
    main()
