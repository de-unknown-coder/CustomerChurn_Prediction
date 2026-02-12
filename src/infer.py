import joblib
import pandas as pd

model = joblib.load("models/churn_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

sample = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": 845.0,
}

df = pd.DataFrame([sample])

X = preprocessor.transform(df)

pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][1]

print("Prediction:", pred)
print("Churn probability:", round(prob, 3))
