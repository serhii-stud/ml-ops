import argparse
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def install_requirements(req_path: str):
    """Install Python dependencies inside the training container (runtime)."""
    if not req_path:
        print("[INFO] No requirements.txt provided. Skipping dependency installation.")
        return
    if not os.path.exists(req_path):
        raise FileNotFoundError(f"requirements.txt not found: {req_path}")

    print(f"[INFO] Installing requirements from: {req_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("[INFO] Requirements installed successfully.")


def parse_args():
    parser = argparse.ArgumentParser()

    # Data files inside SageMaker channels
    parser.add_argument("--train_file", type=str, default="train.parquet")
    parser.add_argument("--test_file", type=str, default="test.parquet")

    # Runtime deps (pyarrow + mlflow)
    parser.add_argument("--requirements", type=str, default=None)

    # Hyperparameters
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=2.0)

    # Data lineage
    parser.add_argument("--data_version", type=str, default="unknown")
    parser.add_argument("--train_s3", type=str, default=None)
    parser.add_argument("--test_s3", type=str, default=None)

    # MLflow
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None)
    parser.add_argument("--mlflow_experiment", type=str, default="banking-support-classifier")
    parser.add_argument("--mlflow_run_name", type=str, default=None)

    parser.add_argument("--min_f1_to_register", type=float, default=0.8)


    # IMPORTANT: ignore unknown SageMaker/system args to avoid ExitCode=2
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[WARN] Ignoring unknown args:", unknown)

    return args


def main():
    print("[DEBUG] argv:", sys.argv)
    args = parse_args()

    # Install deps early (pyarrow for parquet + mlflow)
    install_requirements(args.requirements)

    # Import mlflow AFTER installing requirements
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient


    # SageMaker standard paths
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    test_dir = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    train_path = str(Path(train_dir) / args.train_file)
    test_path = str(Path(test_dir) / args.test_file)

    print("[INFO] Train path:", train_path)
    print("[INFO] Test path:", test_path)
    print("[INFO] Model dir:", model_dir)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # Configure MLflow
    if args.mlflow_tracking_uri:
        print("[INFO] Using MLflow tracking URI:", args.mlflow_tracking_uri)
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    else:
        print("[WARN] mlflow_tracking_uri is not set. MLflow will log locally (not useful).")

    mlflow.set_experiment(args.mlflow_experiment)

    # --- OPTIONAL: sklearn autologging (safe mode) ---
    try:
        mlflow.sklearn.autolog(
            log_models=False,  # модель логируем сами (у тебя mlflow.sklearn.log_model)
            log_input_examples=True,
            silent=True,
        )
        print("[INFO] mlflow.sklearn.autolog enabled (log_models=False).")
    except Exception as e:
        print("[WARN] Failed to enable mlflow autolog (non-fatal):", repr(e))


    # Start MLflow run
    with mlflow.start_run(run_name=args.mlflow_run_name):
        # --- log params / lineage ---
        mlflow.log_param("data_version", args.data_version)
        mlflow.log_param("train_s3", args.train_s3)
        mlflow.log_param("test_s3", args.test_s3)

        mlflow.log_param("train_file", args.train_file)
        mlflow.log_param("test_file", args.test_file)
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("C", args.C)
        mlflow.log_param("model_type", "tfidf+logreg")

        mlflow.set_tag("role", "challenger")              
        mlflow.set_tag("data_version", args.data_version)
        mlflow.set_tag("train_s3", args.train_s3 or "")
        mlflow.set_tag("test_s3", args.test_s3 or "")
        mlflow.set_tag("pipeline", "sagemaker-training")

        # --- load data ---
        df_train = pd.read_parquet(train_path)
        df_test = pd.read_parquet(test_path)

        # --- MLflow data lineage (datasets) ---
        try:
            from mlflow.data import from_pandas

            train_ds = from_pandas(df_train, source=args.train_s3 or train_path, name="train")
            test_ds = from_pandas(df_test,  source=args.test_s3 or test_path,  name="test")

            mlflow.log_input(train_ds, context="training")
            mlflow.log_input(test_ds, context="testing")
            mlflow.log_param("train_rows", int(df_train.shape[0]))
            mlflow.log_param("test_rows", int(df_test.shape[0]))
            mlflow.log_param("train_cols", int(df_train.shape[1]))
            mlflow.log_param("test_cols", int(df_test.shape[1]))
        except Exception as e:
            print("[WARN] MLflow dataset tracking failed (non-fatal):", repr(e))


        X_train = df_train["text"].astype(str).values
        y_train = df_train["category"].astype(str).values

        # Get weights prepared in the Processing Job. Default to 1.0 if the column is missing.
        if "sample_weight" in df_train.columns:
            train_weights = df_train["sample_weight"].values
            print(f"[INFO] Using sample weights. Unique values: {df_train['sample_weight'].unique()}")
        else:
            import numpy as np
            train_weights = np.ones(len(df_train))
            print("[WARN] 'sample_weight' column not found. Defaulting to 1.0.")

        X_test = df_test["text"].astype(str).values
        y_test = df_test["category"].astype(str).values

        # --- train ---
        clf = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(max_features=args.max_features)),
                ("lr", LogisticRegression(C=args.C, max_iter=200, n_jobs=1)),
            ]
        )

        print("[INFO] Training...")
        
        # Using step_name__parameter syntax to pass weights specifically to the LogisticRegression (lr) step.
        clf.fit(X_train, y_train, lr__sample_weight=train_weights)

        # --- eval ---
        print("[INFO] Evaluating...")
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # SageMaker UI metrics (regex parses these)
        print(f"[METRIC] accuracy={acc:.6f}")
        print(f"[METRIC] f1_weighted={f1:.6f}")

        # MLflow metrics
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_weighted", float(f1))

        # --- save model to SM_MODEL_DIR (SageMaker will upload it to S3 automatically) ---
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(clf, model_path)
        print("[INFO] Saved model to:", model_path)

        # Also log model to MLflow artifacts
        mlflow.sklearn.log_model(clf, artifact_path="model")

                # --- register model only if it's good enough ---
        MIN_F1 = float(args.min_f1_to_register or 0.8)
        MODEL_NAME = "banking-support-classifier"

        if float(f1) >= MIN_F1:
            import time

            client = MlflowClient()

            # Get the current MLflow run ID
            run_id = mlflow.active_run().info.run_id

            # URI of the model artifact logged in this run
            # "model" is the artifact_path used in mlflow.sklearn.log_model(...)
            model_uri = f"runs:/{run_id}/model"

            # Register the model in MLflow Model Registry
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_NAME,
            )

            # Optional: wait until the model version becomes READY
            # This avoids race conditions when immediately applying stage or tags
            for _ in range(15):
                version_info = client.get_model_version(MODEL_NAME, mv.version)
                if version_info.status == "READY":
                    break
                time.sleep(1)

            # 1) Move the registered model version to the Staging stage
            # Staging represents a "challenger" candidate for Production
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=mv.version,
                stage="Staging",
            )

            # 2) Add a tag to the MODEL VERSION (not the run)
            # This tag is used later by the promotion job to identify challengers
            client.set_model_version_tag(
                MODEL_NAME,
                mv.version,
                key="role",
                value="challenger",
            )

            # Optional: add run-level tags for easier filtering in Experiments UI
            # These tags do NOT affect the Model Registry
            mlflow.set_tag("role", "challenger")
            mlflow.set_tag("model_name", MODEL_NAME)
            mlflow.set_tag("model_version", str(mv.version))

        else:
            mlflow.set_tag("candidate", "false")
            print(f"[INFO] Model NOT registered (f1={float(f1):.4f} < MIN_F1={MIN_F1:.4f})")


        print("[INFO] MLflow logging finished successfully.")


if __name__ == "__main__":
    main()
    