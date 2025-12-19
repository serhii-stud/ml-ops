import os
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager


MODEL_NAME = "BankingSupportBaseline"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f">>> Connecting to MLflow at {MLFLOW_URI}...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    try:
        # 1. Ищем ВСЕ версии модели (и Production, и None, и Staging)
        # get_latest_versions возвращает список последних версий для КАЖДОГО стейджа
        versions = client.get_latest_versions(MODEL_NAME)

        # 2. Выбираем версию с самым большим номером (самую свежую)
        # Сортируем по номеру версии (int(v.version))
        latest_version = max(versions, key=lambda v: int(v.version))

        print(f">>> Found latest version: v{latest_version.version} (Stage: {latest_version.current_stage})")

        # 3. Загружаем модель
        # models:/ИМЯ/ВЕРСИЯ
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        ml_model["model"] = mlflow.sklearn.load_model(model_uri)

        print(f">>> SUCCESS: Model loaded from {model_uri}")

    except Exception as e:
        # ВАЖНО: Если модель не загрузилась, мы роняем приложение.
        # Docker перезапустит контейнер, и мы попробуем снова.
        print(f"CRITICAL ERROR: Failed to load model. {e}")
        raise e

    yield  # Здесь приложение работает и принимает запросы

    # (Опционально) Очистка ресурсов при выключении
    ml_model.clear()


app = FastAPI(title="Support Triage API", lifespan=lifespan)


class Ticket(BaseModel):
    text: str


@app.post("/predict")
def predict(ticket: Ticket):
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Scikit-learn pipeline ожидает список строк, а не одну строку
        prediction = ml_model["model"].predict([ticket.text])
        return {"category": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")