import pandas as pd
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from preprocess import clean_data


def main():
    # Load dataset
    df = pd.read_csv("data/raw/churn.csv")
    df = clean_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load artifacts
    model = joblib.load("models/churn_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    # Transform
    X_test_proc = preprocessor.transform(X_test)

    # Probabilities
    y_test_bin = y_test.map({"No": 0, "Yes": 1})
    y_probs = model.predict_proba(X_test_proc)[:, 1]

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_probs)
    auc = roc_auc_score(y_test_bin, y_probs)

    print("ROC-AUC:", round(auc, 4))

    # Best threshold (Youden J)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]

    print("Best Threshold:", round(best_threshold, 3))

    # Classification report with tuned threshold
    y_pred = (y_probs >= best_threshold).astype(int)

    print("\nReport (Tuned Threshold):\n")
    print(classification_report(y_test_bin, y_pred))

    # Save threshold
    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": float(best_threshold)}, f)

    print("Saved threshold to models/threshold.json")

    # Plot ROC
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("roc_curve.png")  # 👈 saves image
    plt.show()


main()
