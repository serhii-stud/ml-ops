import os
import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME") or "mlops-project-sm"

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

# Если используешь IAM-роль в SageMaker, можно оставить пустым dict
STORAGE_OPTS = (
    {"key": AWS_KEY, "secret": AWS_SECRET}
    if AWS_KEY and AWS_SECRET
    else {}
)


def read_jsonl_from_s3(path_pattern: str) -> pd.DataFrame:
    """
    Читает все JSONL-файлы по шаблону S3 пути.
    Поддерживает подкаталоги через glob.
    """
    print(f">>> Globbing files: {path_pattern}")

    import s3fs
    fs = s3fs.S3FileSystem()

    # expand wildcard (*)
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


def build_training_dataset(data_version: str | None = None) -> str:
    """
    Собираем датасет для обучения из:
    - data/logs/raw/inference/  (input_text + predicted_category)
    - data/logs/raw/corrections/ (истинный label от оператора)

    Сохраняем:
    - data/processed/runs/<data_version>/train.parquet / test.parquet
    - data/processed/latest/train_latest.parquet / test_latest.parquet

    Возвращаем data_version (нужно будет логировать в MLflow из train-скрипта).
    """
    if not BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME is not set")

    # --- 1. data_version ---
    if data_version is None or data_version == "auto":
        data_version = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

    print(f">>> DATA VERSION: {data_version}")

    # --- 2. Пути к логам ---
    inf_path = f"s3://{BUCKET_NAME}/data/raw/logs/inference/*/*.jsonl"
    print('!!!!' + inf_path)
    corr_path = f"s3://{BUCKET_NAME}/data/raw/logs/corrections/*/*.jsonl"

    # --- 3. Читаем inference ---
    df_inf = read_jsonl_from_s3(inf_path)
    # ожидаем колонки: request_id, input_text, predicted_category, model_name, model_version, timestamp

    # --- 4. Читаем corrections ---
    df_corr = read_jsonl_from_s3(corr_path)
    # возможные варианты названия колонки с истиной:
    # "category" или "corrected_category"
    if "corrected_category" in df_corr.columns:
        df_corr = df_corr.rename(columns={"corrected_category": "true_category"})
    elif "category" in df_corr.columns:
        df_corr = df_corr.rename(columns={"category": "true_category"})
    else:
        raise ValueError("Corrections logs must contain 'category' or 'corrected_category'")

    df_corr = df_corr[["request_id", "true_category"]]

    # --- 5. Джойним по request_id (берём только те, где есть истина) ---
    print(">>> Merging inference + corrections on request_id (inner join)...")
    df = df_inf.merge(df_corr, on="request_id", how="inner", suffixes=("_inf", "_corr"))
    print(f"    Merged rows: {len(df)}")

    if len(df) == 0:
        raise ValueError("No matched rows between inference and corrections by request_id")

    # --- 6. Финальный текст и лейбл ---
    # текст берём из inference.input_text
    df["text"] = df["input_text"]

    # финальный класс = true_category
    df["category"] = df["true_category"]

    # флаг ошибки модели: предсказание != истина
    if "predicted_category" in df.columns:
        df["is_misclassified"] = df["predicted_category"] != df["category"]
    else:
        df["is_misclassified"] = False

    # пример sample_weight: ошибки можно весить сильнее (опционально)
    df["sample_weight"] = df["is_misclassified"].apply(lambda x: 2.0 if x else 1.0)

    # --- 7. Чистим данные ---
    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f">>> After cleaning: {len(df)} samples")

    if len(df["category"].unique()) < 2:
        print("WARNING: only one class present in data, training may fail.")

    # --- 8. Train/Test split ---
    from sklearn.model_selection import train_test_split

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["category"] if df["category"].nunique() > 1 else None,
    )

    print(f"    Train shape: {df_train.shape}")
    print(f"    Test shape:  {df_test.shape}")

    # --- 9. Пути для сохранения (runs + latest) ---
    base_runs = f"s3://{BUCKET_NAME}/data/processed/runs/{data_version}"
    base_latest = f"s3://{BUCKET_NAME}/data/processed/latest"

    train_runs_path = f"{base_runs}/train.parquet"
    test_runs_path = f"{base_runs}/test.parquet"

    train_latest_path = f"{base_latest}/train_latest.parquet"
    test_latest_path = f"{base_latest}/test_latest.parquet"

    # --- 10. Сохраняем версии в runs/<data_version> ---
    print(f">>> Saving versioned datasets to {base_runs} ...")
    df_train.to_parquet(train_runs_path, index=False, storage_options=STORAGE_OPTS)
    df_test.to_parquet(test_runs_path, index=False, storage_options=STORAGE_OPTS)

    # --- 11. Сохраняем алиасы latest/ ---
    print(f">>> Updating latest dataset aliases in {base_latest} ...")
    df_train.to_parquet(train_latest_path, index=False, storage_options=STORAGE_OPTS)
    df_test.to_parquet(test_latest_path, index=False, storage_options=STORAGE_OPTS)

    print(">>> DONE. Data version:", data_version)
    return data_version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_version",
        type=str,
        default="auto",
        help="Version label for this data build (default: auto timestamp)",
    )

    args = parser.parse_args()

    dv = build_training_dataset(data_version=args.data_version)
    print(f"[SUMMARY] Built training data version: {dv}")
