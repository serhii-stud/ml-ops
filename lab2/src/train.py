import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("banking-support-triage")


def train_model():
    print(">>> 1. Loading data from S3...")

    df = pd.read_csv(f"s3://{BUCKET_NAME}/data/raw/banking_train.csv")

    X = df['text']
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(">>> 2. Training Baseline (LogReg)...")
    # Создаем пайплайн: Текст -> Вектор -> Логистическая регрессия
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        # Оценка
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Training Complete. Macro F1: {f1:.4f}")

        # Логирование
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_param("model_type", "LogisticRegression")

        # Сохранение модели в MLflow (она улетит в S3/mlflow/...)
        print(">>> 3. Logging model to MLflow registry...")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="BankingBaseline"
        )
        print("SUCCESS: Model trained and saved.")


if __name__ == "__main__":
    train_model()