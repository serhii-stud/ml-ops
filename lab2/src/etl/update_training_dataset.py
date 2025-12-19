import pandas as pd
import s3fs
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

BUCKET = os.getenv("S3_BUCKET_NAME")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

# Log records older than 6 months are not needed
WINDOW_DAYS = 90


def run_etl():
    # Generate unique version ID at the beginning of the run
    # Format: YYYY-MM-DD_HH-MM-SS (guarantees uniqueness within the day)
    version_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f">>> [ETL] Starting Job. Bucket: {BUCKET}")
    print(f">>> [ETL] Version ID: {version_id}")

    fs = s3fs.S3FileSystem(key=AWS_KEY, secret=AWS_SECRET)
    storage_opts = {"key": AWS_KEY, "secret": AWS_SECRET}

    # ---------------------------------------------------------
    # 1. Read the GOLDEN SET
    # ---------------------------------------------------------
    # We read both train and test from history. Why?
    # Because for future models, the old "test" set is simply high-quality
    # labeled data that we should not lose.
    # The model will be evaluated on a NEW split, so the old split does not matter.

    df_gold_train = pd.DataFrame()
    df_gold_test = pd.DataFrame()

    # Read Historical Data
    try:
        path_train = f"s3://{BUCKET}/data/raw/historical/train.csv"
        path_test = f"s3://{BUCKET}/data/raw/historical/test.csv"

        if fs.exists(path_train):
            df_gold_train = pd.read_csv(path_train, storage_options=storage_opts)

        if fs.exists(path_test):
            df_gold_test = pd.read_csv(path_test, storage_options=storage_opts)

    except Exception as e:
        print(f"   CRITICAL ERROR: Could not read history. Run ingest first! Details: {e}")
        return

    # ---------------------------------------------------------
    # 2. Read LOGS (Fresh data from the sliding window)
    # ---------------------------------------------------------
    print("2. Reading and Filtering Logs...")
    df_logs = pd.DataFrame()
    logs_path = f"{BUCKET}/data/raw/logs/inference/"

    try:
        # Read all .jsonl log files
        log_files = fs.glob(f"{logs_path}*.jsonl")

        if log_files:
            logs_list = []
            cutoff_date = datetime.now() - timedelta(days=WINDOW_DAYS)

            for file_path in log_files:
                with fs.open(file_path) as f:
                    df_chunk = pd.read_json(f, lines=True)

                    # If timestamp exists — filter by window
                    if 'timestamp' in df_chunk.columns:
                        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'])
                        # Keep only fresh records
                        df_chunk = df_chunk[df_chunk['timestamp'] >= cutoff_date]

                    if not df_chunk.empty:
                        logs_list.append(df_chunk)

            if logs_list:
                df_logs = pd.concat(logs_list, ignore_index=True)
                print(f"   Loaded Valid Logs (Last {WINDOW_DAYS} days): {df_logs.shape}")
            else:
                print("   Logs found, but all were too old or empty.")
        else:
            print("   No log files found (Cold Start).")

    except Exception as e:
        print(f"   Warning: Error processing logs ({e}). Proceeding with history only.")

    # ---------------------------------------------------------
    # 3. MERGE
    # ---------------------------------------------------------
    print("3. Merging datasets...")
    full_df = pd.concat([df_gold_train, df_gold_test, df_logs], ignore_index=True)

    if full_df.empty:
        print("Error: Dataset is empty!")
        return

    # Remove duplicate texts, keeping the latest occurence
    full_df.drop_duplicates(subset=['text'], keep='last', inplace=True)

    # ---------------------------------------------------------
    # 4. SPLIT
    # ---------------------------------------------------------
    print("4. Splitting...")
    try:
        train_df, test_df = train_test_split(
            full_df,
            test_size=0.2,
            stratify=full_df['category'],
            random_state=42
        )
    except ValueError:
        # Stratification may fail if some class has only 1 sample — fallback to random split
        train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 5. SAVING (Dual Save Strategy)
    # ---------------------------------------------------------
    print(f"5. Saving version {version_id} to S3...")

    # Folder structure matches s3://project/data/processed/ from documentation
    s3_train_ver = f"s3://{BUCKET}/data/processed/train_{version_id}.parquet"
    s3_test_ver  = f"s3://{BUCKET}/data/processed/test_{version_id}.parquet"

    # Pointers for "LATEST" version used by the Training Service
    s3_train_latest = f"s3://{BUCKET}/data/processed/train_latest.parquet"
    s3_test_latest  = f"s3://{BUCKET}/data/processed/test_latest.parquet"

    # Save versioned files (history)
    train_df.to_parquet(s3_train_ver, storage_options=storage_opts)
    test_df.to_parquet(s3_test_ver, storage_options=storage_opts)

    # Save the LATEST version (overwrites pointer for the training system)
    train_df.to_parquet(s3_train_latest, storage_options=storage_opts)
    test_df.to_parquet(s3_test_latest, storage_options=storage_opts)

    print(f">>> [ETL] Success!")
    print(f"    Saved Version: {s3_train_ver}")
    print(f"    Updated Pointer: {s3_train_latest}")


if __name__ == "__main__":
    if not BUCKET:
        raise ValueError("S3_BUCKET_NAME env variable is missing!")
    run_etl()
