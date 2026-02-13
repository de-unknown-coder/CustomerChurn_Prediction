from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd
import json

app = FastAPI(title="Churn Prediction API")

# Load artifacts
model = joblib.load("models/churn_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

# Load tuned threshold
with open("models/threshold.json") as f:
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
