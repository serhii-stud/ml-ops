import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
import joblib


def install_requirements(req_path: str):
    """Install Python dependencies inside the training container (runtime)."""
    if not req_path:
        return
    if not os.path.exists(req_path):
        raise FileNotFoundError(f"requirements.txt not found: {req_path}")

    print(f"[INFO] Installing requirements from: {req_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("[INFO] Requirements installed successfully.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, default="train.parquet")
    parser.add_argument("--test_file", type=str, default="test.parquet")
    parser.add_argument("--requirements", type=str, default=None)

    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--C", type=float, default=2.0)

    # IMPORTANT: ignore unknown SageMaker/system args to avoid ExitCode=2
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[WARN] Ignoring unknown args:", unknown)

    return args


def main():
    print("[DEBUG] argv:", sys.argv)
    args = parse_args()

    # Install deps early (pyarrow for parquet)
    install_requirements(args.requirements)

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

    # Load data
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # Expect columns from your data_prep: text, category
    X_train = df_train["text"].astype(str).values
    y_train = df_train["category"].astype(str).values

    X_test = df_test["text"].astype(str).values
    y_test = df_test["category"].astype(str).values

    # Baseline model: TF-IDF + Logistic Regression
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

    # Save model artifact
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(clf, model_path)
    print("[INFO] Saved model to:", model_path)


if __name__ == "__main__":
    main()
