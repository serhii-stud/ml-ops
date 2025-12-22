import os
import json
import uuid
import boto3
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

# --- CONFIGURATION ---
app = FastAPI(title="Data Ingestion Service")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
s3_client = boto3.client("s3")

S3_PREFIX = "data/raw/logs/corrections"


# --- DATA MODELS ---
class CorrectionItem(BaseModel):
    request_id: str
    text: str
    category: str
    model_version: str = "unknown"


class CorrectionBatch(BaseModel):
    items: List[CorrectionItem]


# --- ENDPOINTS ---

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_batch(batch: CorrectionBatch):
    if not batch.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch is empty"
        )

    try:
        ingestion_time = datetime.utcnow().isoformat()
        json_lines = []

        for item in batch.items:
            record = item.dict()
            record['ingested_at'] = ingestion_time
            json_lines.append(json.dumps(record, ensure_ascii=False))

        file_content = "\n".join(json_lines)

        batch_id = str(uuid.uuid4())
        date_str = datetime.now().strftime("%Y-%m-%d")

        s3_key = f"{S3_PREFIX}/{date_str}/batch_{batch_id}.jsonl"

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content
        )

        print(f">>> [Ingest] Saved batch of {len(batch.items)} items to {s3_key}")

        return {
            "status": "success",
            "ingested_count": len(batch.items),
            "s3_key": s3_key
        }

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))