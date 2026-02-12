# Binary Classification Project (ROC-AUC + Threshold Selection)

This repository contains a basic machine learning classification project built to understand and implement core evaluation concepts like **ROC Curve** and **AUC Score**.

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

## Why ROC-AUC?

ROC Curve shows how model performance changes across different thresholds.

AUC represents:

> The probability that the model ranks a random positive sample higher than a random negative sample.

This helps measure how well the classifier separates the two classes.

---

## Threshold Selection

Instead of always using the default threshold:
threshold = 0.5
We can choose a better threshold (example: 0.7) depending on the tradeoff between:
 - catching positives (high recall)
 - avoiding false alarms (low false positives)


## Learning Objective

This project is built as a fundamentals-focused step toward stronger machine learning understanding, especially for competitive exams and real-world ML work.