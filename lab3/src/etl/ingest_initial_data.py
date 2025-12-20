import s3fs
import pandas as pd
import os
import sys

# --- CONFIGURATION ---
BUCKET = os.getenv("S3_BUCKET_NAME")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

# Path to local files INSIDE the container (volume mount)
LOCAL_SOURCE_DIR = "/opt/airflow/data_initial"


def ingest():
    print(f">>> [Ingest] Checking S3 bucket: {BUCKET}")

    # Connect to S3
    fs = s3fs.S3FileSystem(key=AWS_KEY, secret=AWS_SECRET)

    # ---------------------------------------------------------
    # 1. UPLOAD HISTORICAL DATA (Train + Test)
    # ---------------------------------------------------------
    files_to_upload = {
        "train.csv": f"{BUCKET}/data/raw/historical/train.csv",
        "test.csv": f"{BUCKET}/data/raw/historical/test.csv"
    }

    # Check that the directory with initial data is mounted into Docker
    if not os.path.exists(LOCAL_SOURCE_DIR):
        print(f"CRITICAL ERROR: Local directory {LOCAL_SOURCE_DIR} not found inside container!")
        print("Did you mount the volume './data_initial:/opt/airflow/data_initial' in docker-compose?")
        sys.exit(1)

    for filename, s3_path in files_to_upload.items():
        local_file_path = os.path.join(LOCAL_SOURCE_DIR, filename)

        # Check if the file already exists in S3 (idempotency)
        if fs.exists(s3_path):
            print(f"   [SKIP] File already exists in S3: {s3_path}")
            continue

        # Check if the file exists locally
        if not os.path.exists(local_file_path):
            print(f"   ERROR: Local file missing: {local_file_path}")
            continue

        print(f"   [COLD START] Uploading {filename}...")
        try:
            df = pd.read_csv(local_file_path)
            with fs.open(s3_path, 'w') as f:
                df.to_csv(f, index=False)
            print(f"   Success: Uploaded to {s3_path}")
        except Exception as e:
            print(f"   ERROR uploading {filename}: {e}")

    # ---------------------------------------------------------
    # 2. CREATE FOLDER STRUCTURE FOR LOGS (Inference + Corrections)
    # ---------------------------------------------------------
    # S3 does not store empty folders, so we create empty `.keep` files

    log_folders = [
        f"{BUCKET}/data/raw/logs/inference/.keep",    # For inference logs
        f"{BUCKET}/data/raw/logs/corrections/.keep"   # For operator corrections
    ]

    for folder_marker in log_folders:
        if not fs.exists(folder_marker):
            try:
                fs.touch(folder_marker)
                print(f"   Created structure: {folder_marker}")
            except Exception as e:
                print(f"   Error creating folder marker {folder_marker}: {e}")
        else:
            print(f"   Structure already exists: {folder_marker}")

    print(">>> [Ingest] Complete.")


if __name__ == "__main__":
    if not BUCKET:
        raise ValueError("S3_BUCKET_NAME env variable is missing")
    ingest()
