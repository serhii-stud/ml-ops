import argparse
import os
import sys
from pathlib import Path

# Импорты сработают, так как SageMaker установит их ДО запуска скрипта
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
import joblib

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        default="train_latest.parquet",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="test_latest.parquet",
    )
    # УДАЛИЛИ --requirements, он больше не нужен
    
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=2.0)

    return parser.parse_args()


def main():
    args = parse_args()

    # УДАЛИЛИ install_requirements() — SageMaker сделает это сам

    # Пути SageMaker
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    test_dir = os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    train_path = str(Path(train_dir) / args.train_file)
    test_path = str(Path(test_dir) / args.test_file)

    print(f"[INFO] Train path: {train_path}")
    print(f"[INFO] Test path: {test_path}")

    # Проверки, что файлы на месте
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    
    # ... (Остальной код загрузки и обучения без изменений) ...
    
    # Load data
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    X_train = df_train["text"].astype(str).values
    y_train = df_train["category"].astype(str).values

    X_test = df_test["text"].astype(str).values
    y_test = df_test["category"].astype(str).values

    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=args.max_features)),
            ("lr", LogisticRegression(C=args.C, max_iter=200, n_jobs=1)),
        ]
    )

    print("[INFO] Training...")
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"[METRIC] accuracy={acc:.6f}")
    print(f"[METRIC] f1_weighted={f1:.6f}")

    # Save model
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))
    print("[INFO] Model saved.")

if __name__ == "__main__":
    main()