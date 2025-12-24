import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
API_URL = "http://0.0.0.0:8000/predict"
TEST_DATA_PATH = "data/raw/test.csv"  # Укажи путь к твоему тестовому файлу
# Если файла test.csv нет, можно попробовать на кусочке train.csv:
# TEST_DATA_PATH = "data/raw/train.csv"

LIMIT_SAMPLES = 200  # Не будем ждать вечность, проверим на 200 примерах


def benchmark():
    print(f"🚀 Starting Benchmark using {TEST_DATA_PATH}...")

    try:
        # Читаем датасет
        df = pd.read_csv(TEST_DATA_PATH)

        # Если датасет огромный, берем случайную выборку
        if len(df) > LIMIT_SAMPLES:
            print(f"✂️ Sampling {LIMIT_SAMPLES} random rows from {len(df)} total.")
            df = df.sample(LIMIT_SAMPLES, random_state=42)

        y_true = []
        y_pred = []
        errors = []

        print(f"⏳ Sending {len(df)} requests...")

        start_global = time.time()

        for index, row in df.iterrows():
            text = row['text']
            true_label = row['category']

            try:
                # Отправляем запрос
                response = requests.post(API_URL, json={"text": text})

                if response.status_code == 200:
                    pred_label = response.json()['category']

                    y_true.append(true_label)
                    y_pred.append(pred_label)

                    # Если ошибка, запоминаем для анализа
                    if pred_label != true_label:
                        errors.append({
                            "text": text,
                            "true": true_label,
                            "pred": pred_label
                        })
                else:
                    print(f"⚠️ API Error: {response.status_code}")

            except Exception as e:
                print(f"🚨 Connection Error: {e}")
                break

            # Небольшая пауза не нужна, мы хотим проверить скорость работы тоже,
            # но для локального докера можно оставить микро-паузу
            # time.sleep(0.01)

        total_time = time.time() - start_global

        # --- ОТЧЕТ ---
        print("\n" + "=" * 40)
        print("📊 BENCHMARK RESULTS")
        print("=" * 40)

        acc = accuracy_score(y_true, y_pred)
        print(f"✅ Accuracy: {acc:.2%}")
        print(f"⏱️ Avg Latency: {total_time / len(df):.4f} sec/req")

        print("\n❌ Top 5 Mistakes:")
        for err in errors[:5]:
            print(f"   Input: '{err['text'][:50]}...'")
            print(f"   Expected: {err['true']}")
            print(f"   Got:      {err['pred']}")
            print("-" * 20)

    except FileNotFoundError:
        print(f"❌ Error: File {TEST_DATA_PATH} not found.")
        print("Please check the path or put a CSV file with 'text' and 'category' columns.")


if __name__ == "__main__":
    benchmark()