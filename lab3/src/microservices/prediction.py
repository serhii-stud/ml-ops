import os
import json
import uuid
import boto3
import asyncio
import mlflow.sklearn
from datetime import datetime
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- КОНФИГУРАЦИЯ ---
MODEL_NAME = "BankingSupportBaseline"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Настройки батчинга
FLUSH_INTERVAL_SECONDS = 60  # Сбрасываем логи раз в минуту
LOG_BUFFER = []  # Глобальный буфер в памяти

ml_model = {}
s3_client = boto3.client("s3")


async def flush_logs_periodically():
    """Фоновая задача: раз в N секунд отправляет накопленные логи в S3."""
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
            await flush_buffer_to_s3()
        except asyncio.CancelledError:
            # Если сервис останавливают, выходим из цикла
            break
        except Exception as e:
            print(f"ERROR in background logger: {e}")


async def flush_buffer_to_s3():
    """Синхронная отправка данных из буфера в S3."""
    global LOG_BUFFER

    if not LOG_BUFFER:
        return  # Буфер пуст, экономим запрос к S3

    # 1. Забираем данные и очищаем глобальный буфер (атомарно для event loop)
    # Копируем ссылки в локальную переменную, а глобальную чистим
    current_batch = LOG_BUFFER[:]
    LOG_BUFFER.clear()

    count = len(current_batch)
    print(f">>> Flushing {count} logs to S3...")

    try:
        # 2. Формируем тело файла (NDJSON)
        # Каждый элемент — это JSON-строка + перенос строки
        body = "\n".join([json.dumps(record, ensure_ascii=False) for record in current_batch])

        # 3. Генерируем имя файла: logs/inference/YYYY-MM-DD/batch_HHMMSS_uuid.jsonl
        now = datetime.utcnow()
        date_folder = now.strftime("%Y-%m-%d")
        file_name = f"batch_{now.strftime('%H%M%S')}_{uuid.uuid4()}.jsonl"
        key = f"data/raw/logs/inference/{date_folder}/{file_name}"

        # 4. Отправляем в S3 (в потоке, чтобы не блочить, используем run_in_executor если нужно,
        # но boto3 быстрый, для простоты оставим так)
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=body
        )
        print(f"✅ Saved batch to {key}")

    except Exception as e:
        print(f"CRITICAL: Failed to write logs to S3. Data lost! Error: {e}")
        # В реальном проде тут можно было бы вернуть данные обратно в LOG_BUFFER


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- ЗАПУСК ---
    print(f">>> Connecting to MLflow at {MLFLOW_URI}...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(MODEL_NAME)
        latest_version = max(versions, key=lambda v: int(v.version))
        print(f">>> Found latest version: v{latest_version.version}")

        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        ml_model["model"] = mlflow.sklearn.load_model(model_uri)
        ml_model["version"] = latest_version.version
        print(f">>> SUCCESS: Model loaded")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. {e}")
        # Не падаем, чтобы контейнер жил, но работать не будем

    # Запускаем фоновую задачу логирования
    logger_task = asyncio.create_task(flush_logs_periodically())

    yield

    # --- ОСТАНОВКА ---
    print(">>> Shutting down... Flushing remaining logs.")
    # Отменяем фоновую задачу
    logger_task.cancel()
    # Финальный сброс остатков перед смертью
    await flush_buffer_to_s3()
    ml_model.clear()


app = FastAPI(title="Support Triage API", lifespan=lifespan)


class Ticket(BaseModel):
    text: str


def buffer_prediction(request_id: str, ticket_text: str, prediction: str, model_version: str):
    """Просто добавляет запись в список (в памяти)."""
    timestamp = datetime.utcnow().isoformat()

    log_entry = {
        "request_id": request_id,
        "timestamp": timestamp,
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "input_text": ticket_text,
        "predicted_category": prediction
    }

    LOG_BUFFER.append(log_entry)
    # Если вдруг буфер переполнился (например, 1000 записей), можно форсированно сбросить,
    # но пока оставим простую логику по таймеру.


@app.post("/predict")
def predict(ticket: Ticket, request: Request):
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Получаем или генерируем ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

    try:
        # Предикт
        prediction_array = ml_model["model"].predict([ticket.text])
        predicted_category = prediction_array[0]
        current_version = ml_model.get("version", "unknown")

        # Логирование (теперь это мгновенно, т.к. пишем в память)
        buffer_prediction(
            request_id=request_id,
            ticket_text=ticket.text,
            prediction=predicted_category,
            model_version=current_version
        )

        return {
            "request_id": request_id,
            "category": predicted_category,
            "model_version": current_version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")