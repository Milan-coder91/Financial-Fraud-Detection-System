# Financial Fraud Detection System

## Project Overview
A Machine Learning-based application designed to identify potentially fraudulent financial transactions using transaction data.

## Features
- Dataset Overview and Analysis
- Fraud Analytics Dashboard
- Machine Learning Model Comparison
- Fraud Transaction Prediction
- Adjustable Fraud Detection Threshold
- Interactive Streamlit Dashboard

## Dataset
- Total Transactions: 5,000
- Fraudulent Transactions: 482
- Legitimate Transactions: 4,518
- Fraud Rate: 9.64%

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest

Preliminary selected model: Logistic Regression based on PR-AUC.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Status
Interactive dashboard and initial model evaluation completed. Further validation is required before production use.
