from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
import json

app = FastAPI(title="Churn Prediction API")

# Load artifacts
import os

# Absolute base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct absolute paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.json")

# Load artifacts safely
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(THRESHOLD_PATH) as f:
    THRESHOLD = json.load(f)["threshold"]


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int

    PhoneService: str
    MultipleLines: str
    InternetService: str

    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str

    StreamingTV: str
    StreamingMovies: str

    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Churn API is running"}


@app.post("/predict")
def predict(data: CustomerInput):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)

    # Probability of churn (Yes)
    prob = model.predict_proba(X)[0][1]

    # Apply tuned threshold
    pred = "Yes" if prob >= THRESHOLD else "No"

    return {
        "churn": pred,
        "probability": round(float(prob), 3),
        "threshold_used": round(float(THRESHOLD), 3)
    }
