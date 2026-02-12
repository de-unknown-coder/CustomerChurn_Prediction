import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from preprocess import clean_data, make_preprocessor


df = pd.read_csv("data/raw/churn.csv")
df = clean_data(df)

x = df.drop("Churn", axis=1)
y = df["Churn"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = make_preprocessor(x_train)

x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

preds = model.predict(x_test)
acc = accuracy_score(y_test, preds)

y_test_bin = y_test.map({"No": 0, "Yes": 1})
probs = model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test_bin, probs)

print("Accuracy:", round(acc, 4))
print("AUC:", round(auc, 4))

joblib.dump(model, "models/churn_model.joblib")
joblib.dump(preprocessor, "models/preprocessor.joblib")

print("Saved model + preprocessor")
