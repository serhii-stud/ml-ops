import pandas as pd
import s3fs
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

BUCKET = os.getenv("S3_BUCKET_NAME")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
WINDOW_DAYS = 180  # Данные логов старше 6 месяцев нам не нужны


def run_etl():
    print(f">>> [ETL] Starting Job. Bucket: {BUCKET}")

    # Настраиваем доступ к S3
    fs = s3fs.S3FileSystem(key=AWS_KEY, secret=AWS_SECRET)
    storage_opts = {"key": AWS_KEY, "secret": AWS_SECRET}

    # ---------------------------------------------------------
    # 1. Читаем GOLDEN SET (Исторические данные)
    # ---------------------------------------------------------
    # Мы берем И train И test из истории. Почему?
    # Потому что для будущей модели старый "test" - это просто хорошие,
    # качественно размеченные данные, которые грех терять.
    # Мы проверим модель на НОВОМ split-е.

    df_gold_train = pd.DataFrame()
    df_gold_test = pd.DataFrame()

    print("1. Reading Historical Data (Golden Set)...")
    try:
        path_train = f"s3://{BUCKET}/data/raw/historical/train.csv"
        path_test = f"s3://{BUCKET}/data/raw/historical/test.csv"

        if fs.exists(path_train):
            df_gold_train = pd.read_csv(path_train, storage_options=storage_opts)

        if fs.exists(path_test):
            df_gold_test = pd.read_csv(path_test, storage_options=storage_opts)

        print(f"   Loaded History: Train {df_gold_train.shape}, Test {df_gold_test.shape}")

    except Exception as e:
        print(f"   CRITICAL ERROR: Could not read history. Run ingest first! Details: {e}")
        return

    # ---------------------------------------------------------
    # 2. Читаем LOGS (Свежие данные с окна)
    # ---------------------------------------------------------
    print("2. Reading and Filtering Logs...")
    df_logs = pd.DataFrame()
    logs_path = f"{BUCKET}/data/raw/logs/inference/"

    try:
        # Ищем все файлы .jsonl
        log_files = fs.glob(f"{logs_path}*.jsonl")

        if log_files:
            logs_list = []
            cutoff_date = datetime.now() - timedelta(days=WINDOW_DAYS)

            for file_path in log_files:
                with fs.open(file_path) as f:
                    # Читаем JSONL
                    df_chunk = pd.read_json(f, lines=True)

                    # Если в логах есть дата - фильтруем
                    if 'timestamp' in df_chunk.columns:
                        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'])
                        # Оставляем только свежее
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
    # 3. MERGE (Слияние)
    # ---------------------------------------------------------
    print("3. Merging datasets...")
    # Объединяем всё, что нашли
    full_df = pd.concat([df_gold_train, df_gold_test, df_logs], ignore_index=True)

    # Проверка: если данных нет вообще
    if full_df.empty:
        print("Error: Dataset is empty! Nothing to save.")
        return

    # Дедупликация: Если текст повторяется, оставляем последний (свежий) вариант
    before_dedup = len(full_df)
    full_df.drop_duplicates(subset=['text'], keep='last', inplace=True)
    print(f"   Merged size: {before_dedup} -> {len(full_df)} (Deduplicated)")

    # ---------------------------------------------------------
    # 4. SPLIT (Разбиение)
    # ---------------------------------------------------------
    print("4. Creating new Train/Test Split (80/20)...")

    try:
        # Stratify=full_df['category'] гарантирует, что в тесте будут все классы
        train_df, test_df = train_test_split(
            full_df,
            test_size=0.2,
            stratify=full_df['category'],
            random_state=42
        )
    except ValueError:
        # Если какой-то класс встречается всего 1 раз, стратификация упадет.
        # В этом случае делаем обычный рандомный сплит.
        print("   Warning: Rare classes detected. Stratification failed. Using random split.")
        train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)

    print(f"   Final Shapes -> Train: {train_df.shape}, Test: {test_df.shape}")

    # ---------------------------------------------------------
    # 5. СОХРАНЕНИЕ (Parquet)
    # ---------------------------------------------------------
    print("5. Saving to S3...")

    # Сохраняем как 'latest', чтобы модель всегда знала, где брать актуальные данные
    s3_train_path = f"s3://{BUCKET}/data/processed/train_latest.parquet"
    s3_test_path = f"s3://{BUCKET}/data/processed/test_latest.parquet"

    train_df.to_parquet(s3_train_path, storage_options=storage_opts)
    test_df.to_parquet(s3_test_path, storage_options=storage_opts)

    print(f">>> [ETL] Success! Data ready at: {s3_train_path}")


if __name__ == "__main__":
    if not BUCKET:
        raise ValueError("S3_BUCKET_NAME env variable is missing!")
    run_etl()