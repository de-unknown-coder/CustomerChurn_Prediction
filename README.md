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

## ROC Curve

We evaluated the model using the ROC curve, which plots:

>TPR (True Positive Rate / Recall)
 Fraction of churners correctly identified

>FPR (False Positive Rate)
 Fraction of non-churners incorrectly flagged

The ROC-AUC score achieved:
ROC-AUC = 0.836

## Selecting an Optimal Threshold
![ROC Curve](assets/roc_curve.png)
Instead of using the default threshold (0.5), we selected a threshold using Youden’s J statistic:

J=TPR−FPR

Best threshold found:

Threshold ≈ 0.239

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

