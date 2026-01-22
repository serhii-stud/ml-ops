# evaluation/evaluate.py
import argparse
import os
import subprocess
import sys
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def install_requirements(req_path: str):
    """Install dependencies like pyarrow / mlflow at runtime (if provided)."""
    if req_path and os.path.exists(req_path):
        print(f"[INFO] Installing requirements from: {req_path}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("[INFO] Requirements installed.")


def parse_data_key(data_version: str) -> str:
    """
    From: banking-prep-2026-01-07-14-52-41-c0e0b2
    Get:  2026-01-07-14-52-41-c0e0b2
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-[a-f0-9]+)$", data_version)
    if not m:
        raise ValueError(f"Cannot parse data key from data_version: {data_version}")
    return m.group(1)


def parse_args():
    p = argparse.ArgumentParser()

    # --- Data lineage key ---
    p.add_argument("--data_version", type=str, required=True,
                   help="e.g. banking-prep-2026-01-07-14-52-41-c0e0b2")

    # --- Input data ---
    p.add_argument("--test_file", type=str, default="test.parquet",
                   help="File name inside the mounted test channel")

    # --- MLflow ---
    p.add_argument("--mlflow_tracking_uri", type=str, required=True)
    p.add_argument("--mlflow_experiment", type=str, default="banking-support-classifier")

    # Model artifact inside MLflow run
    p.add_argument("--model_artifact_path", type=str, default="model",
                   help="Artifact folder/path within the train run (e.g. 'model')")
    p.add_argument("--model_file", type=str, default="model.joblib",
                   help="Model filename inside artifact_path (e.g. model.joblib)")

    # Optional: if you want to narrow down by training run name (your 'banking-train-...')
    p.add_argument("--train_run_name", type=str, default=None,
                   help="Optional: filter candidates whose mlflow.runName contains this string")

    # Dependencies
    p.add_argument("--requirements", type=str, default=None,
                   help="Path to requirements.txt in the processing container")

    # How many latest runs to scan when fallback to runName-contains search
    p.add_argument("--scan_last_runs", type=int, default=200)

    return p.parse_args()


def find_train_run_id(client, experiment_id: str, data_version: str,
                      train_run_name: str, scan_last_runs: int) -> str:
    """
    Try (A) params.data_version == data_version
    If not found -> (B) scan latest runs and pick where runName contains data_key
    Additionally optional contains train_run_name (if provided).
    """
    data_key = parse_data_key(data_version)

    # A) exact param match (works if train logs param "data_version")
    try:
        filter_a = f"params.data_version = '{data_version}'"
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_a,
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs:
            rid = runs[0].info.run_id
            rn = (runs[0].data.tags or {}).get("mlflow.runName", "")
            print(f"[INFO] Found train run by params.data_version. run_id={rid}, run_name={rn}")
            return rid
    except Exception as e:
        print(f"[WARN] Search by params.data_version failed (will fallback). Reason: {e}")

    # B) fallback: scan latest runs and pick by runName contains data_key (+ optional train_run_name)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="",
        order_by=["attributes.start_time DESC"],
        max_results=scan_last_runs,
    )

    for r in runs:
        rn = (r.data.tags or {}).get("mlflow.runName", "")
        if data_key in rn:
            if train_run_name and (train_run_name not in rn):
                continue
            print(f"[INFO] Found train run by runName contains data_key. run_id={r.info.run_id}, run_name={rn}")
            return r.info.run_id

    raise RuntimeError(
        f"Cannot find train run for data_version={data_version} (data_key={data_key}). "
        f"Tried params.data_version and scanning last {scan_last_runs} runs."
    )


def main():
    args = parse_args()
    install_requirements(args.requirements)

    import mlflow
    from mlflow.tracking import MlflowClient

    # --- MLflow setup ---
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    client = MlflowClient()

    exp = client.get_experiment_by_name(args.mlflow_experiment)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {args.mlflow_experiment}")

    # --- Find the train run for this data_version ---
    train_run_id = find_train_run_id(
        client=client,
        experiment_id=exp.experiment_id,
        data_version=args.data_version,
        train_run_name=args.train_run_name,
        scan_last_runs=args.scan_last_runs,
    )

    # --- Download model artifacts from MLflow ---
    download_root = "/opt/ml/processing/model_download"
    Path(download_root).mkdir(parents=True, exist_ok=True)

    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=train_run_id,
        artifact_path=args.model_artifact_path,
        dst_path=download_root,
    )
    model_path = os.path.join(downloaded_path, args.model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Downloaded artifact_path='{args.model_artifact_path}' to: {downloaded_path}\n"
            f"Check --model_artifact_path / --model_file values."
        )

    print(f"[INFO] Loading model from: {model_path}")
    clf = joblib.load(model_path)

    # --- Load test data ---
    test_dir = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/processing/input/test")
    test_path = os.path.join(test_dir, args.test_file)
    print(f"[INFO] Loading test data from: {test_path}")

    df_test = pd.read_parquet(test_path)

    # --- Predict ---
    if "text" not in df_test.columns or "category" not in df_test.columns:
        raise ValueError("Test dataframe must contain columns: 'text' and 'category'")

    X_test = df_test["text"].astype(str).values
    y_true = df_test["category"].astype(str).values
    y_pred = clf.predict(X_test)

    df_test["prediction"] = y_pred

    # --- Metrics ---
    metrics: dict[str, float] = {}
    metrics["overall_acc"] = float(accuracy_score(y_true, y_pred))
    metrics["overall_f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))

    if "source" in df_test.columns:
        df_golden = df_test[df_test["source"] == "golden"]
        if not df_golden.empty:
            metrics["golden_acc"] = float(accuracy_score(df_golden["category"], df_golden["prediction"]))
            metrics["golden_f1_weighted"] = float(f1_score(df_golden["category"], df_golden["prediction"], average="weighted"))
        else:
            print("[WARN] No golden rows found (source == 'golden').")

        df_feedback = df_test[df_test["source"] == "feedback"]
        if not df_feedback.empty:
            metrics["feedback_acc"] = float(accuracy_score(df_feedback["category"], df_feedback["prediction"]))
            metrics["feedback_f1_weighted"] = float(f1_score(df_feedback["category"], df_feedback["prediction"], average="weighted"))
        else:
            print("[WARN] No feedback rows found (source == 'feedback').")
    else:
        print("[WARN] Column 'source' not found; slice metrics will be skipped.")

    # --- Log to MLflow: separate eval-run linked to train_run_id ---
    data_key = parse_data_key(args.data_version)
    eval_run_name = f"banking-eval-{data_key}"

    with mlflow.start_run(run_name=eval_run_name):
        mlflow.set_tag("pipeline_step", "evaluation")
        mlflow.set_tag("data_version", args.data_version)
        mlflow.set_tag("data_key", data_key)
        mlflow.set_tag("parent_train_run_id", train_run_id)
        if args.train_run_name:
            mlflow.set_tag("parent_train_run_name_hint", args.train_run_name)

        print("[INFO] Logging evaluation metrics to MLflow...")
        for k, v in metrics.items():
            print(f"[METRIC] {k}: {v:.6f}")
            mlflow.log_metric(k, v)

        # Optional: log a small report artifact
        report_path = "/opt/ml/processing/eval_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"data_version: {args.data_version}\n")
            f.write(f"parent_train_run_id: {train_run_id}\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        mlflow.log_artifact(report_path, artifact_path="evaluation")

    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
