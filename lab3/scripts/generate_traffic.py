import requests
import time
import random
import sys

# --- КОНФИГУРАЦИЯ ---
API_URL = "http://0.0.0.0:8000/predict"
TOTAL_REQUESTS = 100  # Сколько логов сгенерировать
MIN_DELAY = 0.1  # Мин. задержка (сек)
MAX_DELAY = 0.5  # Макс. задержка (сек)

# Примеры запросов пользователей (разные интенты)
USER_QUERIES = [
    # Card Arrival
    "Where is my card?",
    "I ordered a card last week",
    "Track my card delivery status",
    "When will the credit card arrive?",

    # Lost/Stolen
    "I lost my wallet with my card",
    "Help, my card was stolen",
    "Block my card immediately please",
    "I cannot find my visa card",

    # Balance
    "What is my current balance?",
    "How much money do I have left?",
    "Show me my account total",
    "Balance inquiry",

    # Payments
    "My payment was declined at the shop",
    "Why can't I buy this coffee?",
    "Transaction rejected",
    "Card not working",

    # General / Other
    "Hello, are you a bot?",
    "I want to speak to a human",
    "What are your working hours?",
    "Do you offer loans?"
]


def generate_traffic():
    print(f"🚀 Starting traffic generation to {API_URL}...")
    print(f"🎯 Goal: {TOTAL_REQUESTS} requests")
    print("-" * 40)

    success_count = 0
    fail_count = 0

    for i in range(1, TOTAL_REQUESTS + 1):
        # 1. Выбираем случайный текст
        text = random.choice(USER_QUERIES)

        # 2. Формируем payload (как ожидает твой FastAPI: class Ticket(BaseModel): text: str)
        payload = {"text": text}

        try:
            # 3. Отправляем запрос
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            elapsed = time.time() - start_time

            # 4. Обрабатываем ответ
            if response.status_code == 200:
                data = response.json()
                req_id = data.get("request_id", "NO_ID")
                category = data.get("category", "UNKNOWN")
                print(f"[{i}/{TOTAL_REQUESTS}] ✅ OK ({elapsed:.2f}s) | ID: {req_id} | Pred: {category}")
                success_count += 1
            else:
                print(f"[{i}/{TOTAL_REQUESTS}] ❌ Error {response.status_code}: {response.text}")
                fail_count += 1

        except requests.exceptions.ConnectionError:
            print(f"[{i}/{TOTAL_REQUESTS}] 🚨 Connection Refused! Is the service running at {API_URL}?")
            fail_count += 1
        except Exception as e:
            print(f"[{i}/{TOTAL_REQUESTS}] ⚠️ Unexpected error: {e}")
            fail_count += 1

        # 5. Пауза, чтобы не "положить" сервис
        sleep_time = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(sleep_time)

    print("-" * 40)
    print(f"🏁 Done! Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    generate_traffic()