# Financial Fraud Detection System

## Project Overview
The Financial Fraud Detection System is a Machine Learning-based application designed to identify potentially fraudulent financial transactions.

It analyzes transaction details and predicts the probability of fraud using trained machine learning models.

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

The preliminary selected model is Logistic Regression based on PR-AUC.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

## How to Run Locally

1. Clone the repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Project Status
The interactive dashboard and initial model evaluation are completed. Further validation is required before production use.
