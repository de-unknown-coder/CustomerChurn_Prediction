import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"


def clean_data(df):
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])
    return df


def make_preprocessor(X):
    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
