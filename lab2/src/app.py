from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import os
import pandas as pd

app = FastAPI(title="Support Triage API")

model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        model_name = "BankingSupportBaseline"
        print(f"Loading model: {model_name}...")

        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model {model_uri} loaded successfully!")
    except Exception as e:
        print(f"WARNING: Could not load model. Error: {e}")


class Ticket(BaseModel):
    text: str


@app.post("/predict")
def predict(ticket: Ticket):
    if not model:
        return {"error": "Model is not loaded yet"}

    prediction = model.predict([ticket.text])
    return {"category": prediction[0]}