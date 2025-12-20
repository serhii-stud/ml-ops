import os
import json
import uuid
import boto3
import mlflow.sklearn
from datetime import datetime
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

MODEL_NAME = "BankingSupportBaseline"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

ml_model = {}
s3_client = boto3.client("s3")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f">>> Connecting to MLflow at {MLFLOW_URI}...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    try:
        versions = client.get_latest_versions(MODEL_NAME)
        latest_version = max(versions, key=lambda v: int(v.version))

        print(f">>> Found latest version: v{latest_version.version} (Stage: {latest_version.current_stage})")

        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        ml_model["model"] = mlflow.sklearn.load_model(model_uri)
        ml_model["version"] = latest_version.version

        print(f">>> SUCCESS: Model loaded from {model_uri}")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. {e}")
        raise e

    yield
    ml_model.clear()


app = FastAPI(title="Support Triage API", lifespan=lifespan)


class Ticket(BaseModel):
    text: str


def log_prediction_to_s3(ticket_text: str, prediction: str, model_version: str):
    """
    Logs data in JSONL format (single line + \n).
    S3 does not support appending to files, so we write separate objects
    that can be easily concatenated later.
    """
    try:
        request_id = str(uuid.uuid4())
        # Use UTC for logs - this is standard practice
        timestamp = datetime.utcnow().isoformat()

        log_data = {
            "request_id": request_id,
            "timestamp": timestamp,
            "model_name": MODEL_NAME,
            "model_version": model_version,
            "input_text": ticket_text,
            "predicted_category": prediction
        }

        # 1. Convert to JSON string (no indentation!) and add a newline character
        jsonl_line = json.dumps(log_data, ensure_ascii=False) + "\n"

        # 2. Construct the path: logs/YYYY-MM-DD/uuid.jsonl
        date_folder = datetime.now().strftime("%Y-%m-%d")
        file_key = f"data/raw/logs/inference/{date_folder}/{request_id}.jsonl"

        # 3. Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=file_key,
            Body=jsonl_line
        )
        # Optional: print what was saved (helps with debugging via docker logs)
        print(f"Log S3: {file_key}")

    except Exception as e:
        print(f"WARNING: Failed to save log to S3: {e}")


@app.post("/predict")
def predict(ticket: Ticket):
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prediction_array = ml_model["model"].predict([ticket.text])
        predicted_category = prediction_array[0]

        log_prediction_to_s3(
            ticket_text=ticket.text,
            prediction=predicted_category,
            model_version=ml_model.get("version", "unknown")
        )

        return {"category": predicted_category}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")