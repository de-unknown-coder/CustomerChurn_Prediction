import pandas as pd
from src.preprocess import clean_data

def test_clean_data_drops_customerid():
    df = pd.DataFrame({
        "customerID": ["0001"],
        "TotalCharges": ["100"],
        "Churn": ["No"]
    })

    cleaned = clean_data(df)

    assert "customerID" not in cleaned.columns

