from fastapi import FastAPI
from pydantic import BaseModel

import joblib
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Load saved artifacts
model = joblib.load("models/churn_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")


# ---- Input Schema ----
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
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Transform with saved preprocessor
    X = preprocessor.transform(df)

    # Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "churn": pred,
        "probability": round(float(prob), 3)
    }
