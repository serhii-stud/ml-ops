import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    """Parse CLI args passed by the SageMaker Processing Job launcher."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_version",
        type=str,
        default="auto",
        help="Version label for this data build (default: auto).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/opt/ml/processing/input",
        help="Local path where SageMaker mounts input data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/opt/ml/processing/output",
        help="Local path where SageMaker expects outputs to be written.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Unique run identifier. If provided, used as data_version for traceability.",
    )

    return parser.parse_args()


def read_jsonl_files(file_paths):
    """Read multiple JSONL files from local filesystem into a single DataFrame."""
    if not file_paths:
        raise FileNotFoundError("No JSONL files found for the given pattern.")

    dfs = []
    for fp in file_paths:
        print(f"    Reading {fp}")
        df = pd.read_json(fp, lines=True)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    print(f"    Total rows loaded: {len(result)}")
    return result


def build_training_dataset(data_version: str, input_dir: str, output_dir: str) -> str:
    """
    Builds train/test datasets from local inputs mounted by SageMaker.

    Expected input structure (mounted from S3 via ProcessingInput):
      <input_dir>/
        historical/
          train.csv
          test.csv
        logs/
          inference/<any_subfolders>/*.jsonl
          corrections/<any_subfolders>/*.jsonl

    Output structure (written locally; SageMaker uploads output_dir to S3 via ProcessingOutput):
      <output_dir>/
        train.parquet
        test.parquet
        latest/
          train_latest.parquet
          test_latest.parquet

    Notes:
    - This script does NOT use S3 paths internally.
    - S3 -> local mount is handled by ProcessingInput.
    - local -> S3 upload is handled by ProcessingOutput.
    - Parquet requires pyarrow (installed via requirements.txt + dependencies=[...]).
    """

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> INPUT_DIR:  {in_dir}")
    print(f">>> OUTPUT_DIR: {out_dir}")
    print(f">>> DATA VERSION: {data_version}")

    # ======================================================
    # 1. Load GOLDEN dataset (historical labeled data)
    # ======================================================
    golden_train_path = in_dir / "historical" / "train.csv"
    golden_test_path = in_dir / "historical" / "test.csv"

    if not golden_train_path.exists():
        raise FileNotFoundError(f"GOLDEN train not found: {golden_train_path}")
    if not golden_test_path.exists():
        raise FileNotFoundError(f"GOLDEN test not found: {golden_test_path}")

    print(f">>> Loading GOLDEN train from: {golden_train_path}")
    golden_train = pd.read_csv(golden_train_path)
    print(f"    GOLDEN train shape: {golden_train.shape}")

    print(f">>> Loading GOLDEN test  from: {golden_test_path}")
    golden_test = pd.read_csv(golden_test_path)
    print(f"    GOLDEN test shape:  {golden_test.shape}")

    # Ensure consistent structure
    golden_train = golden_train[["text", "category"]].copy()
    golden_test = golden_test[["text", "category"]].copy()

    golden_train["source"] = "golden"
    golden_test["source"] = "golden"
    golden_train["sample_weight"] = 1.0
    golden_test["sample_weight"] = 1.0
    golden_train["is_misclassified"] = False
    golden_test["is_misclassified"] = False

    # ======================================================
    # 2. Load FEEDBACK data (local logs)
    # ======================================================
    inf_dir = in_dir / "logs" / "inference"
    corr_dir = in_dir / "logs" / "corrections"

    print(f">>> FEEDBACK inference dir:   {inf_dir}")
    print(f">>> FEEDBACK corrections dir: {corr_dir}")

    try:
        inf_files = list(inf_dir.rglob("*.jsonl")) if inf_dir.exists() else []
        corr_files = list(corr_dir.rglob("*.jsonl")) if corr_dir.exists() else []

        print(f"    Found inference JSONL files:   {len(inf_files)}")
        print(f"    Found corrections JSONL files: {len(corr_files)}")

        if not inf_files or not corr_files:
            raise FileNotFoundError("Feedback logs not found (inference or corrections missing).")

        df_inf = read_jsonl_files(inf_files)
        df_corr = read_jsonl_files(corr_files)

        # Normalize corrections column name
        if "corrected_category" in df_corr.columns:
            df_corr = df_corr.rename(columns={"corrected_category": "true_category"})
        elif "category" in df_corr.columns:
            df_corr = df_corr.rename(columns={"category": "true_category"})
        else:
            raise ValueError("Corrections logs must contain 'category' or 'corrected_category'")

        df_corr = df_corr[["request_id", "true_category"]]

        # Merge inference with corrections
        print(">>> Merging inference + corrections on request_id (inner join)...")
        df_fb = df_inf.merge(df_corr, on="request_id", how="inner")
        print(f"    FEEDBACK merged rows: {len(df_fb)}")

        if len(df_fb) == 0:
            print("WARNING: No matched rows in feedback logs — GOLDEN only will be used.")
            df_fb = pd.DataFrame(
                columns=["text", "category", "source", "sample_weight", "is_misclassified"]
            )
        else:
            df_fb["text"] = df_fb["input_text"]
            df_fb["category"] = df_fb["true_category"]

            if "predicted_category" in df_fb.columns:
                df_fb["is_misclassified"] = df_fb["predicted_category"] != df_fb["category"]
            else:
                df_fb["is_misclassified"] = False

            # Optionally amplify the weight of misclassified samples
            df_fb["sample_weight"] = df_fb["is_misclassified"].apply(lambda x: 2.0 if x else 1.0)
            df_fb["source"] = "feedback"

            # Clean dataset
            df_fb = df_fb.dropna(subset=["text", "category"])
            df_fb = df_fb[df_fb["text"].astype(str).str.strip() != ""]
            print(f">>> FEEDBACK after cleaning: {len(df_fb)} samples")

    except FileNotFoundError as e:
        print(f"WARNING: feedback logs not found: {e}")
        df_fb = pd.DataFrame(
            columns=["text", "category", "source", "sample_weight", "is_misclassified"]
        )

    # ======================================================
    # 3. Build final train/test datasets
    # ======================================================
    if len(df_fb) == 0:
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

        df_train = pd.concat([golden_train, fb_train], ignore_index=True)
        df_test = pd.concat([golden_test, fb_test], ignore_index=True)

    print(f">>> FINAL train shape: {df_train.shape}")
    print(f">>> FINAL test shape:  {df_test.shape}")
    print("    Train source breakdown:")
    print(df_train["source"].value_counts(dropna=False))
    print("    Test source breakdown:")
    print(df_test["source"].value_counts(dropna=False))

    # ======================================================
    # 4. Save datasets locally as Parquet
    # ======================================================
    latest_dir = out_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    train_out = out_dir / "train.parquet"
    test_out = out_dir / "test.parquet"
    train_latest_out = latest_dir / "train_latest.parquet"
    test_latest_out = latest_dir / "test_latest.parquet"

    print(f">>> Saving train to: {train_out}")
    df_train.to_parquet(train_out, index=False)

    print(f">>> Saving test to:  {test_out}")
    df_test.to_parquet(test_out, index=False)

    print(f">>> Updating latest aliases in: {latest_dir}")
    df_train.to_parquet(train_latest_out, index=False)
    df_test.to_parquet(test_latest_out, index=False)

    print(">>> DONE. Data version:", data_version)
    return data_version


if __name__ == "__main__":
    args = parse_args()

    # Use run_id as the effective version if provided (best for traceability)
    effective_version = args.run_id if args.run_id else args.data_version

    # If still "auto", generate a timestamp (Python 3.9 compatible)
    if effective_version is None or effective_version == "auto":
        effective_version = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    dv = build_training_dataset(
        data_version=effective_version,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(f"[SUMMARY] Built training data version: {dv}")
