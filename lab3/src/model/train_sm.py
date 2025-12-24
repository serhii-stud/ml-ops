import os
import argparse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

from dotenv import load_dotenv

load_dotenv()


def get_or_create_experiment(experiment_name: str, artifact_root: str) -> str:
    print(f">>> Configuring Experiment: {experiment_name} -> {artifact_root}")
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_root
        )
        print(f">>> Created new experiment ID: {experiment_id}")
        return experiment_id
    except MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise RuntimeError(f"Experiment {experiment_name} not found and cannot be created")
        print(f">>> Experiment exists. ID: {experiment.experiment_id}")
        return experiment.experiment_id


def get_current_production_f1(client: MlflowClient, model_name: str, metric_name: str = "f1_macro"):
    """
    Берём текущую champion-модель из Production-стадии
    и читаем метрику из run-а. Если нет прод-модели — возвращаем None.
    """
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except MlflowException:
        return None, None

    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if not prod_versions:
        print(">>> No Production model yet. This run will be the first champion.")
        return None, None

    prod_version = sorted(prod_versions, key=lambda v: int(v.version))[-1]
    prod_run_id = prod_version.run_id
    run = client.get_run(prod_run_id)
    metrics = run.data.metrics
    f1_prod = metrics.get(metric_name)

    print(f">>> Current Production model: v{prod_version.version}, run_id={prod_run_id}, {metric_name}={f1_prod}")
    return f1_prod, prod_version


def train_model(
    mlflow_uri: str,
    bucket_name: str,
    experiment_name: str,
    registered_model_name: str,
    f1_improvement_threshold: float = 0.0,
):
    print(">>> [TRAIN] Initializing...")

    if not mlflow_uri or not bucket_name:
        raise ValueError("MLFLOW_TRACKING_URI or S3_BUCKET_NAME not set!")

    mlflow.set_tracking_uri(mlflow_uri)

    artifact_root = f"s3://{bucket_name}/mlflow_experiments"
    experiment_id = get_or_create_experiment(experiment_name, artifact_root)
    mlflow.set_experiment(experiment_id=experiment_id)

    storage_opts = {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }

    # ---------------------------------------------------------
    # 1. Load Datasets train and test
    # ---------------------------------------------------------
    print(">>> 1. Loading pre-split data from S3 (latest version)...")

    train_path = f"s3://{bucket_name}/data/processed/train_latest.parquet"
    test_path = f"s3://{bucket_name}/data/processed/test_latest.parquet"

    try:
        df_train = pd.read_parquet(train_path, storage_options=storage_opts)
        df_test = pd.read_parquet(test_path, storage_options=storage_opts)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load data from S3. Did ETL run? Error: {e}")
        return

    print(f"    Train shape: {df_train.shape}")
    print(f"    Test shape:  {df_test.shape}")

    X_train = df_train["text"]
    y_train = df_train["category"]

    X_test = df_test["text"]
    y_test = df_test["category"]

    # ---------------------------------------------------------
    # 2. Training with class weights
    # ---------------------------------------------------------
    print(">>> 2. Training Baseline (LogReg with class_weight='balanced')...")

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=1000)),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced"  # ← балансировка классов
            )),
        ]
    )

    client = MlflowClient()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f">>> Started MLflow run: {run_id}")

        pipeline.fit(X_train, y_train)

        # ---------------------------------------------------------
        # 3. Evaluation
        # ---------------------------------------------------------
        y_pred = pipeline.predict(X_test)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        print(f"    Training Complete. Macro F1 on Test Set: {f1_macro:.4f}")

        # Можно дополнительно сохранить отчёт по классам
        cls_report = classification_report(y_test, y_pred, output_dict=False)
        print(cls_report)

        # ---------------------------------------------------------
        # 4. Log metrics & params in MLflow
        # ---------------------------------------------------------
        print(">>> 3. Logging metrics and artifacts to MLflow...")

        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("data_source_train", train_path)
        mlflow.log_param("data_source_test", test_path)
        mlflow.log_param("tfidf_max_features", 1000)
        mlflow.log_param("clf_C", 1.0)
        mlflow.log_param("clf_class_weight", "balanced")

        # ---------------------------------------------------------
        # 5. Register model (as challenger)
        # ---------------------------------------------------------
        print(">>> 4. Registering model in MLflow Model Registry...")

        mv = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        # mv — это ModelVersion, полученный из log_model
        model_version = mv.version
        print(f">>> Registered as {registered_model_name} v{model_version}")

        # ставим тег challenger/metrics на саму версию
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="role",
            value="challenger",
        )
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version,
            key="f1_macro",
            value=str(f1_macro),
        )

        # ---------------------------------------------------------
        # 6. Champion/Challenger Logic
        # ---------------------------------------------------------
        print(">>> 5. Champion/Challenger comparison...")

        f1_prod, prod_version = get_current_production_f1(client, registered_model_name)

        promote_to_prod = False
        if f1_prod is None:
            # Прод-модели ещё нет — делаем текущую champion сразу
            promote_to_prod = True
        else:
            if f1_macro >= f1_prod + f1_improvement_threshold:
                promote_to_prod = True

        if promote_to_prod:
            print(">>> New model is better (or first). Promoting to Production...")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version,
                stage="Production",
                archive_existing_versions=True,
            )
            client.set_model_version_tag(
                name=registered_model_name,
                version=model_version,
                key="role",
                value="champion",
            )
        else:
            print(">>> New model did NOT beat current Production. Keeping as challenger only.")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version,
                stage="Staging",
            )

        print(">>> SUCCESS: Training, evaluation, registry update complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Эти аргументы удобно пробрасывать из SageMaker Training Job
    parser.add_argument("--mlflow_uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--bucket_name", type=str, default=os.getenv("S3_BUCKET_NAME"))
    parser.add_argument("--experiment_name", type=str, default="banking-support-classifier-v2")
    parser.add_argument("--registered_model_name", type=str, default="BankingSupportBaseline")
    parser.add_argument("--f1_improvement_threshold", type=float, default=0.0)

    args = parser.parse_args()

    train_model(
        mlflow_uri=args.mlflow_uri,
        bucket_name=args.bucket_name,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        f1_improvement_threshold=args.f1_improvement_threshold,
    )
