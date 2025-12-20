import pandas as pd
import s3fs
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")


def ingest_data():
    print(">>> 1. Downloading Banking77 dataset...")
    # Скачиваем с GitHub (как в твоем дизайне)
    url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    df = pd.read_csv(url)
    print(f"Dataset downloaded: {df.shape}")

    print(f">>> 2. Uploading to AWS S3: {BUCKET_NAME}...")

    # Используем s3fs для удобной записи
    fs = s3fs.S3FileSystem(key=AWS_KEY, secret=AWS_SECRET)

    s3_path = f"{BUCKET_NAME}/data/raw/banking_train.csv"

    with fs.open(s3_path, 'w') as f:
        df.to_csv(f, index=False)

    print(f"SUCCESS: Data saved to s3://{s3_path}")


if __name__ == "__main__":
    if not BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME is not set in .env")
    ingest_data()