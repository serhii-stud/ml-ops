import os
import json
import uuid
import boto3
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- CONFIGURATION ---
MODEL_NAME = "N/A (Sagemeker Endpoint)"
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# --- SageMaker Endpoint ---
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "banking-support-classifier-prod")
SAGEMAKER_CONTENT_TYPE = os.getenv("SAGEMAKER_CONTENT_TYPE", "application/json")

# Batching settings
FLUSH_INTERVAL_SECONDS = 60  # Flush logs once a minute
LOG_BUFFER = []  # Global in-memory buffer

ml_model = {}
s3_client = boto3.client("s3")

# SageMaker runtime client (data-plane)
sm_runtime = boto3.client("sagemaker-runtime")


async def flush_logs_periodically():
    """Background task: flushes accumulated logs to S3 every N seconds."""
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL_SECONDS)
            await flush_buffer_to_s3()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"ERROR in background logger: {e}")


async def flush_buffer_to_s3():
    """Synchronously uploads data from buffer to S3."""
    global LOG_BUFFER

    if not LOG_BUFFER:
        return  # Buffer is empty, save an S3 request

    current_batch = LOG_BUFFER[:]
    LOG_BUFFER.clear()

    count = len(current_batch)
    print(f">>> Flushing {count} logs to S3...")

    try:
        body = "\n".join([json.dumps(record, ensure_ascii=False) for record in current_batch])

        now = datetime.utcnow()
        date_folder = now.strftime("%Y-%m-%d")
        file_name = f"batch_{now.strftime('%H%M%S')}_{uuid.uuid4()}.jsonl"
        key = f"data/raw/logs/inference/{date_folder}/{file_name}"

        if not S3_BUCKET_NAME:
            print("CRITICAL: S3_BUCKET_NAME is not set. Logs will be dropped.")
            return

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=key,
            Body=body
        )
        print(f"✅ Saved batch to {key}")

    except Exception as e:
        print(f"CRITICAL: Failed to write logs to S3. Data lost! Error: {e}")


def call_sagemaker_endpoint(text: str) -> str:
    """
    Calls SageMaker endpoint and returns predicted category as string.

    Endpoint expects:
      {
        "dataframe_split": {
          "columns": ["text"],
          "data": [["..."]]
        }
      }

    Endpoint returns:
      {"predictions": ["country_support"]}
    """
    endpoint_name = ml_model.get("endpoint_name")
    if not endpoint_name:
        raise RuntimeError("Endpoint is not configured")

    payload = {
        "inputs": [text]
    }

    resp = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=SAGEMAKER_CONTENT_TYPE,
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )

    raw = resp["Body"].read().decode("utf-8")

    try:
        data = json.loads(raw)
    except Exception:
        raise RuntimeError(f"Non-JSON response from endpoint: {raw[:300]}")

    if isinstance(data, dict) and "predictions" in data and isinstance(data["predictions"], list) and data["predictions"]:
        return str(data["predictions"][0])

    # Fallbacks (in case handler changes)
    if isinstance(data, dict):
        if "category" in data:
            return str(data["category"])
        if "prediction" in data:
            return str(data["prediction"])

    raise RuntimeError(f"Unexpected endpoint response: {data}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    if not SAGEMAKER_ENDPOINT_NAME:
        print("CRITICAL ERROR: SAGEMAKER_ENDPOINT_NAME is not set. Model calls will not work.")
    else:
        ml_model["endpoint_name"] = SAGEMAKER_ENDPOINT_NAME
        ml_model["version"] = "endpoint"  # keep contract field
        print(f">>> SUCCESS: Endpoint configured: {SAGEMAKER_ENDPOINT_NAME}")

    # Start background logging task
    logger_task = asyncio.create_task(flush_logs_periodically())

    yield

    # --- SHUTDOWN ---
    print(">>> Shutting down... Flushing remaining logs.")
    logger_task.cancel()
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


@app.post("/predict")
def predict(ticket: Ticket, request: Request):
    if "endpoint_name" not in ml_model:
        raise HTTPException(status_code=503, detail="Endpoint not configured")

    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    try:
        predicted_category = call_sagemaker_endpoint(ticket.text)
        current_version = ml_model.get("version", "unknown")

        buffer_prediction(
            request_id=request_id,
            ticket_text=ticket.text,
            prediction=predicted_category,
            model_version=current_version
        )

        # Keep existing response contract
        return {
            "request_id": request_id,
            "category": predicted_category,
            "model_version": current_version
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
