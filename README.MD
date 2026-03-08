# Credit Decision Support System

## Overview

This project develops a **Decision Support System for Credit Risk Assessment** designed for financial institutions and cooperatives.

The objective is to transition from **traditional accounting-based scoring methods** to a **statistical learning model** capable of improving credit risk detection, reducing loan defaults, and accelerating credit file analysis.

By leveraging financial ratios and statistical modeling, the system assists credit committees in **identifying high-risk borrowers before loan approval**.

---

## Objectives

* Improve **credit risk evaluation**
* Reduce **loan default rates**
* Optimize **credit file processing time**
* Provide **data-driven insights** to credit committees

---

## Methodology

### 1. Data Engineering

Financial data is processed to generate key indicators commonly used in credit analysis.

Examples of engineered variables include:

* **Solvency Ratio**
* **Debt Ratio**
* **Liquidity Indicators**
* **Profitability Metrics**

These ratios transform raw accounting data into meaningful features for risk evaluation.

---

### 2. Statistical Modeling

The model uses **Logistic Regression**, a widely used technique in credit risk modeling, to estimate the probability of borrower default.

Steps include:

1. Data preprocessing and normalization
2. Feature selection based on financial relevance
3. Model training using historical credit data
4. Validation using **AUC (Area Under the ROC Curve)** to measure discriminatory power

---

### 3. Model Performance

The model demonstrates strong predictive capability:

* **Accuracy:** ~90%
* Effective separation between **low-risk and high-risk borrowers**
* Robust validation through ROC-AUC evaluation

---

## Practical Use Case

The system is designed as a **decision-support tool** for credit committees.

It helps to:

* Automatically **flag high-risk loan applications**
* Prioritize **manual review for borderline cases**
* Improve the **efficiency and reliability of credit approval processes**

This allows financial institutions to maintain **better portfolio quality while accelerating loan processing**.

---

## Technologies

* Python
* Pandas
* Scikit-learn
* NumPy
* Jupyter Notebook

---

## Project Structure

```
data/
    raw_data.csv
    processed_data.csv

notebooks/
    data_preparation.ipynb
    model_training.ipynb

src/
    feature_engineering.py
    model.py
    evaluation.py
```

---

## Future Improvements

* Integration of **machine learning models** (Random Forest, Gradient Boosting)
* Deployment as a **web-based credit scoring tool**
* Integration with **financial databases and APIs**
* Real-time credit risk dashboards

---

## Author

**Kemil Alberto Sowah**

Finance & Technology enthusiast working on the intersection of **financial analysis, data science, and software development**.
