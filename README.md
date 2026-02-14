# Customer Churn Prediction API (ROC-AUC + Threshold Tuning)

This repository contains a basic machine learning classification project built to understand and implement core evaluation concepts like **ROC Curve** and **AUC Score**.
An end-to-end churn prediction system using Logistic Regression, deployed with FastAPI and threshold tuning based on ROC analysis.

The goal of this project is not complex engineering, but strong fundamentals in how classification models are evaluated beyond simple accuracy.

---

## Key Concepts Covered

- Binary classification (Yes/No prediction)
- Logistic Regression model
- Confusion Matrix (TP, FP, FN, TN)
- True Positive Rate (TPR) and False Positive Rate (FPR)
- ROC Curve visualization
- AUC Score as a ranking metric
- Threshold tuning instead of blindly using 0.5

---
## Project Structure
## Project Structure

```text
CustomerChurn_Prediction/
│
├── app.py                  # FastAPI inference server
├── requirements.txt
├── README.md
│
├── src/
│   ├── preprocess.py       # Data cleaning + encoding pipeline
│   ├── train.py            # Model training script
│   ├── evaluate.py         # ROC curve + threshold tuning
│   └── infer.py            # Local prediction test
│
├── models/
│   ├── churn_model.joblib
│   ├── preprocessor.joblib
│   └── threshold.json
│
├── assets/
│   └── roc_curve.png
│
├── examples/
│   ├── sample_request.json
│   └── sample_response.json
│
└── tests/
    ├── test_preprocess.py
    ├── test_model.py
    └── test_api.py
```

## Installation

Clone the repo:

```bash
git clone https://github.com/de-unknown-coder/CustomerChurn_Prediction.git
cd CustomerChurn_Prediction
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows
pip install -r requirements.txt
```
---

After installation:

```md
## Training the Model

Run:

```bash
python src/train.py
```

Then:

```md
## ROC Evaluation + Threshold Selection

Run:

```bash
python src/evaluate.py
```
This generates:

ROC curve plot

![ROC Curve](assets/roc_curve.png)

Threshold.json for deployment
Add:

```md
## FastAPI Deployment

Start server:

```bash
uvicorn app:app --reload
```
Open Swagger UI:
http://127.0.0.1:8000/docs

```md
## Example API Request

```json
{
  "tenure": 60,
  "Contract": "Two year",
  "MonthlyCharges": 45.0,
  "TotalCharges": 2700.0,
  ...
}
Example Response
{
    "churn": "No",
  "probability": 0.152,
  "threshold_used": 0.239
}
```
## Example Request/Response Files

Sample API payloads are provided in the `examples/` folder:

- `examples/sample_request.json`
- `examples/sample_response.json`

You can use them directly in Postman or Swagger UI.
---
## ROC Curve 
We evaluated the model using the ROC curve, which plots:
    
>TPR (True Positive Rate / Recall)
  Fraction of churners correctly identified
    
>FPR (False Positive Rate)
  Fraction of non-churners incorrectly flagged
    
The ROC-AUC score achieved:
ROC-AUC = 0.836
---

## Selecting an Optimal Threshold

Instead of using the default threshold (0.5), we selected a threshold using Youden’s J statistic:

J=TPR−FPR

Best threshold found:

Threshold ≈ 0.239
---

## Running Unit Tests

Basic unit tests are included to validate:

- preprocessing logic
- model probability output
- API availability

Run all tests with:

```bash
pytest
```
Expected output:
3 passed
---

## Why ROC Curve

ROC Curve shows how model performance changes across different thresholds.

AUC represents:
> The probability that the model ranks a random positive sample higher than a random negative sample.

This helps measure how well the classifier separates the two classes.
---

## Threshold Tuning (Business-Aware Classification)

The churn model outputs a probability:

𝑃(Churn =𝑌𝑒𝑠)

By default, Logistic Regression uses a threshold of 0.5:

probability ≥ 0.5 → predict churn

probability < 0.5 → predict no churn

However, churn prediction is an imbalanced classification problem, where missing churners can be more costly than false alarms.

## Threshold Selection

Instead of always using the default threshold:
threshold = 0.5
We can choose a better threshold (example: 0.7) depending on the tradeoff between:
 - catching positives (high recall)
 - avoiding false alarms (low false positives)

## Tradeoff: Recall vs Precision

Lowering the threshold increases churn detection:

Churn Recall increased from ~0.53 → ~0.83

But it also increases false positives:

More non-churn customers are flagged as churn

This is a business decision:

>If losing a customer is expensive → prefer high recall

>If retention offers are costly → prefer higher precision

>The tuned threshold is applied during API inference.


## Learning Objective

This project is built as a fundamentals-focused step toward stronger machine learning understanding, especially for competitive exams and real-world ML work.
---
## Future Improvements

- Dockerize deployment
- Deploy API on Render/AWS
- Add monitoring for model drift

