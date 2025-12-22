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

# --- CONFIGURATION ---
MODEL_NAME = "BankingSupportBaseline"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Batching settings
FLUSH_INTERVAL_SECONDS = 60  # Flush logs once a minute
LOG_BUFFER = []  # Global in-memory buffer

ml_model = {}
s3_client = boto3.client("s3")


async def flush_logs_periodically():
    """Background task: flushes accumulated logs to S3 every N seconds."""
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
            await flush_buffer_to_s3()
        except asyncio.CancelledError:
            # If the service is stopping, exit the loop
            break
        except Exception as e:
            print(f"ERROR in background logger: {e}")


async def flush_buffer_to_s3():
    """Synchronously uploads data from buffer to S3."""
    global LOG_BUFFER

    if not LOG_BUFFER:
        return  # Buffer is empty, save an S3 request

    # 1. Retrieve data and clear global buffer (atomic for event loop)
    # Copy references to local variable, clear the global one
    current_batch = LOG_BUFFER[:]
    LOG_BUFFER.clear()

    count = len(current_batch)
    print(f">>> Flushing {count} logs to S3...")

    try:
        # 2. Create file body (NDJSON)
        # Each element is a JSON string + newline
        body = "\n".join([json.dumps(record, ensure_ascii=False) for record in current_batch])

        # 3. Generate filename: logs/inference/YYYY-MM-DD/batch_HHMMSS_uuid.jsonl
        now = datetime.utcnow()
        date_folder = now.strftime("%Y-%m-%d")
        file_name = f"batch_{now.strftime('%H%M%S')}_{uuid.uuid4()}.jsonl"
        key = f"data/raw/logs/inference/{date_folder}/{file_name}"

        # 4. Upload to S3
        # Note: put_object is blocking, but since it runs once/min in background,
        # it's acceptable for this scale. For high load, consider run_in_executor.
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=body
        )
        print(f"✅ Saved batch to {key}")

    except Exception as e:
        print(f"CRITICAL: Failed to write logs to S3. Data lost! Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
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
        # Don't crash so the container stays alive, but service won't work

    # Start background logging task
    logger_task = asyncio.create_task(flush_logs_periodically())

    yield

    # --- SHUTDOWN ---
    print(">>> Shutting down... Flushing remaining logs.")
    # Cancel background task
    logger_task.cancel()
    # Final flush of remaining logs before death
    await flush_buffer_to_s3()
    ml_model.clear()


app = FastAPI(title="Support Triage API", lifespan=lifespan)


class Ticket(BaseModel):
    text: str


def buffer_prediction(request_id: str, ticket_text: str, prediction: str, model_version: str):
    """Adds a record to the list (in memory)."""
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
    # If buffer overflows (e.g., >10k items), we could force flush here,
    # but the timer logic is sufficient for now.


@app.post("/predict")
def predict(ticket: Ticket, request: Request):
    if "model" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get or generate ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

    try:
        # Predict
        prediction_array = ml_model["model"].predict([ticket.text])
        predicted_category = prediction_array[0]
        current_version = ml_model.get("version", "unknown")

        # Logging (now instant, as we write to memory)
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