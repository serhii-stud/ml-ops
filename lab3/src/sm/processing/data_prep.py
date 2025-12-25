import os
import argparse
from datetime import datetime


import pandas as pd

# --------------------------------------------------------------------
# S3 configuration
# --------------------------------------------------------------------
BUCKET_NAME = os.getenv("S3_BUCKET_NAME") or "mlops-project-sm"

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

# When running inside SageMaker Studio, an empty dict is usually enough
# — IAM execution role will handle permissions.
STORAGE_OPTS = (
    {"key": AWS_KEY, "secret": AWS_SECRET}
    if AWS_KEY and AWS_SECRET
    else {}
)


# --------------------------------------------------------------------
# Utility: read JSONL files from S3 based on a glob pattern
# --------------------------------------------------------------------
def read_jsonl_from_s3(path_pattern: str) -> pd.DataFrame:
    """
    Reads all JSONL files matching the S3 glob pattern.
    Supports subfolder patterns like:
        s3://bucket/data/raw/logs/inference/*/*.jsonl
    """
    print(f">>> Globbing files: {path_pattern}")

    import s3fs
    fs = s3fs.S3FileSystem()

    files = fs.glob(path_pattern)
    print(f"    Found {len(files)} files")

    if not files:
        raise FileNotFoundError(f"No files matched pattern: {path_pattern}")

    dfs = []
    for file in files:
        print(f"    Reading {file}")
        with fs.open(file, "r") as f:
            df = pd.read_json(f, lines=True)
            dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    print(f"    Total rows loaded: {len(result)}")
    return result


# --------------------------------------------------------------------
# Main function: build training and test datasets
# --------------------------------------------------------------------
def build_training_dataset(data_version: str = None) -> str:
    """
    Builds train/test datasets from two data sources:

    1) GOLDEN dataset (historical labeled data):
       s3://<bucket>/data/raw/historical/train.csv
       s3://<bucket>/data/raw/historical/test.csv

    2) FEEDBACK dataset (feedback loop from production):
       s3://<bucket>/data/raw/logs/inference/*/*.jsonl
       s3://<bucket>/data/raw/logs/corrections/*/*.jsonl

       - inference logs contain: request_id, input_text, predicted_category, ...
       - corrections logs contain: request_id + ground-truth label (category/corrected_category)

    Output structure:
    - data/processed/runs/<data_version>/train.parquet / test.parquet
    - data/processed/latest/train_latest.parquet / test_latest.parquet

    Where:
    - train = GOLDEN_train ∪ FEEDBACK_train
    - test  = GOLDEN_test  ∪ FEEDBACK_test
    """

    if not BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME is not set")

    # ---------------- 1. data_version ----------------
    if data_version is None or data_version == "auto":
        data_version = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")



    print(f">>> DATA VERSION: {data_version}")

    # ======================================================
    # 2. Load GOLDEN dataset (historical labeled data)
    # ======================================================
    golden_train_path = f"s3://{BUCKET_NAME}/data/raw/historical/train.csv"
    golden_test_path = f"s3://{BUCKET_NAME}/data/raw/historical/test.csv"

    print(f">>> Loading GOLDEN train from: {golden_train_path}")
    golden_train = pd.read_csv(golden_train_path)

    print(f"    GOLDEN train shape: {golden_train.shape}")

    print(f">>> Loading GOLDEN test  from: {golden_test_path}")
    golden_test = pd.read_csv(golden_test_path)
    print(f"    GOLDEN test shape:  {golden_test.shape}")

    # Ensure consistent structure: text + category
    # Assumes CSV already contains these columns.
    golden_train = golden_train[["text", "category"]].copy()
    golden_test = golden_test[["text", "category"]].copy()

    golden_train["source"] = "golden"
    golden_test["source"] = "golden"
    golden_train["sample_weight"] = 1.0
    golden_test["sample_weight"] = 1.0
    golden_train["is_misclassified"] = False
    golden_test["is_misclassified"] = False

    # ======================================================
    # 3. Load FEEDBACK data (inference logs + corrections)
    # ======================================================
    inf_path = f"s3://{BUCKET_NAME}/data/raw/logs/inference/*/*.jsonl"
    corr_path = f"s3://{BUCKET_NAME}/data/raw/logs/corrections/*/*.jsonl"

    print(f">>> FEEDBACK inference path:  {inf_path}")
    print(f">>> FEEDBACK corrections path:{corr_path}")

    try:
        df_inf = read_jsonl_from_s3(inf_path)
        df_corr = read_jsonl_from_s3(corr_path)

        # --- Normalize corrections column name ---
        if "corrected_category" in df_corr.columns:
            df_corr = df_corr.rename(columns={"corrected_category": "true_category"})
        elif "category" in df_corr.columns:
            df_corr = df_corr.rename(columns={"category": "true_category"})
        else:
            raise ValueError(
                "Corrections logs must contain 'category' or 'corrected_category'"
            )

        df_corr = df_corr[["request_id", "true_category"]]

        # --- Merge inference with corrections ---
        print(">>> Merging inference + corrections on request_id (inner join)...")
        df_fb = df_inf.merge(
            df_corr,
            on="request_id",
            how="inner",
            suffixes=("_inf", "_corr"),
        )
        print(f"    FEEDBACK merged rows: {len(df_fb)}")

        if len(df_fb) == 0:
            print("WARNING: No matched rows in feedback logs — GOLDEN only will be used.")
            df_fb = pd.DataFrame(
                columns=[
                    "text",
                    "category",
                    "source",
                    "sample_weight",
                    "is_misclassified",
                ]
            )
        else:
            # Extract final text and true label
            df_fb["text"] = df_fb["input_text"]
            df_fb["category"] = df_fb["true_category"]

            # Misclassification flag
            if "predicted_category" in df_fb.columns:
                df_fb["is_misclassified"] = (
                    df_fb["predicted_category"] != df_fb["category"]
                )
            else:
                df_fb["is_misclassified"] = False

            # Optionally amplify the weight of misclassified samples
            df_fb["sample_weight"] = df_fb["is_misclassified"].apply(
                lambda x: 2.0 if x else 1.0
            )

            df_fb["source"] = "feedback"

            # Clean dataset
            df_fb = df_fb.dropna(subset=["text", "category"])
            df_fb = df_fb[df_fb["text"].astype(str).str.strip() != ""]
            print(f">>> FEEDBACK after cleaning: {len(df_fb)} samples")

    except FileNotFoundError as e:
        print(f"WARNING: feedback logs not found: {e}")
        df_fb = pd.DataFrame(
            columns=[
                "text",
                "category",
                "source",
                "sample_weight",
                "is_misclassified",
            ]
        )

    # ======================================================
    # 4. Build final train/test datasets
    # ======================================================
    from sklearn.model_selection import train_test_split

    if len(df_fb) == 0:
        # No feedback → use GOLDEN dataset only
        print(">>> No FEEDBACK data, using GOLDEN only.")
        df_train = golden_train.copy()
        df_test = golden_test.copy()

    else:
        print(">>> Splitting FEEDBACK into train/test...")

        n_samples_fb = len(df_fb)
        n_classes_fb = df_fb["category"].nunique()
        test_size_fb = 0.2
        planned_test_size = int(n_samples_fb * test_size_fb)

        # If too few samples → disable stratification
        if planned_test_size < n_classes_fb:
            print(
                "WARNING: not enough FEEDBACK samples per class for stratified split. "
                f"planned_test_size={planned_test_size}, n_classes={n_classes_fb}. "
                "Using random split without stratify."
            )
            stratify_arg = None
        else:
            stratify_arg = df_fb["category"]

        fb_train, fb_test = train_test_split(
            df_fb,
            test_size=test_size_fb,
            random_state=42,
            stratify=stratify_arg,
        )

        print(f"    FEEDBACK train shape: {fb_train.shape}")
        print(f"    FEEDBACK test  shape: {fb_test.shape}")

        # Combine GOLDEN + FEEDBACK
        df_train = pd.concat([golden_train, fb_train], ignore_index=True)
        df_test = pd.concat([golden_test, fb_test], ignore_index=True)

    print(f">>> FINAL train shape: {df_train.shape}")
    print(f">>> FINAL test shape:  {df_test.shape}")
    print("    Train source breakdown:")
    print(df_train["source"].value_counts(dropna=False))
    print("    Test source breakdown:")
    print(df_test["source"].value_counts(dropna=False))

    # ======================================================
    # 5. Save versioned and "latest" datasets to S3
    # ======================================================
    base_runs = f"s3://{BUCKET_NAME}/data/processed/runs/{data_version}"
    base_latest = f"s3://{BUCKET_NAME}/data/processed/latest"

    train_runs_path = f"{base_runs}/train.parquet"
    test_runs_path = f"{base_runs}/test.parquet"

    train_latest_path = f"{base_latest}/train_latest.parquet"
    test_latest_path = f"{base_latest}/test_latest.parquet"

    print(f">>> Saving versioned datasets to {base_runs} ...")
    df_train.to_parquet(train_runs_path, index=False)
    df_test.to_parquet(test_runs_path, index=False)

    print(f">>> Updating latest dataset aliases in {base_latest} ...")
    df_train.to_parquet(train_latest_path, index=False)
    df_test.to_parquet(test_latest_path, index=False)

    print(">>> DONE. Data version:", data_version)
    return data_version


# --------------------------------------------------------------------
# CLI entry point — convenient for local run or Processing Job
# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_version",
        type=str,
        default="auto",
        help="Version label for this data build (default: auto timestamp)",
    )

    # Accept SageMaker Processing Job arguments (even if we do not use input/output dirs yet)
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="SageMaker-mounted input directory (not used in S3-native mode).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="SageMaker-mounted output directory (not used in S3-native mode).",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique run identifier passed by the launcher for traceability.",
    )

    args = parser.parse_args()

    # Prefer run_id as the data version to keep S3 outputs aligned with the job run id
    effective_version = args.run_id if args.run_id else args.data_version

    dv = build_training_dataset(data_version=effective_version)
    print(f"[SUMMARY] Built training data version: {dv}")
