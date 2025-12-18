import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("banking-support-classifier")


def train_model():
    print(">>> [TRAIN] Initializing...")

    storage_opts = {"key": AWS_KEY, "secret": AWS_SECRET}

    # ---------------------------------------------------------
    # 1. Load Datasets train and test
    # ---------------------------------------------------------
    print(">>> 1. Loading pre-split data from S3 (latest version)...")

    try:
        train_path = f"s3://{BUCKET_NAME}/data/processed/train_latest.parquet"
        test_path = f"s3://{BUCKET_NAME}/data/processed/test_latest.parquet"

        df_train = pd.read_parquet(train_path, storage_options=storage_opts)
        df_test = pd.read_parquet(test_path, storage_options=storage_opts)

        print(f"    Train shape: {df_train.shape}")
        print(f"    Test shape:  {df_test.shape}")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load data from S3. Did ETL run? Error: {e}")
        return

    X_train = df_train['text']
    y_train = df_train['category']

    X_test = df_test['text']
    y_test = df_test['category']

    # ---------------------------------------------------------
    # 2. Learn model
    # ---------------------------------------------------------
    print(">>> 2. Training Baseline (LogReg)...")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        # ---------------------------------------------------------
        # 3. Evaluation
        # ---------------------------------------------------------
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"    Training Complete. Macro F1 on Test Set: {f1:.4f}")

        # ---------------------------------------------------------
        # 4. Log in MLflow
        # ---------------------------------------------------------
        print(">>> 3. Logging metrics and artifacts to MLflow...")

        mlflow.log_metric("f1_macro", f1)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("data_source", "train_latest.parquet")

        # Save model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="",
            registered_model_name="BankingSupportBaseline"
        )
        print(">>> SUCCESS: Model trained, evaluated, and pushed to Registry.")


if __name__ == "__main__":
    if not BUCKET_NAME:
        raise ValueError("Environment variables (S3_BUCKET_NAME) are missing!")
    train_model()