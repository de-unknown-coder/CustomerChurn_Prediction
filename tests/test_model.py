import joblib
import pandas as pd

def test_model_outputs_valid_probability():
    model = joblib.load("models/churn_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    sample = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 50.0
    }])

    X = preprocessor.transform(sample)
    prob = model.predict_proba(X)[0][1]

    assert 0.0 <= prob <= 1.0
