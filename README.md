# 🛡️ Financial Fraud Detection System

### Machine Learning-Based Financial Fraud Detection & Analytics Dashboard

<p align="center">
  An interactive Machine Learning application designed to detect potentially fraudulent financial transactions using Python, Scikit-learn, and Streamlit.
</p>

<p align="center">
  <a href="https://github.com/Milan-coder91/Financial-Fraud-Detection-System">GitHub Repository</a> |
  <a href="https://financial-fraud-detection-system-2ymqxhtffqc3jckqhwc499.streamlit.app/">Live Streamlit Dashboard</a>
</p>

---

## 🔗 Project Links

- **GitHub Repository:** https://github.com/Milan-coder91/Financial-Fraud-Detection-System
- **Live Streamlit Application:** https://financial-fraud-detection-system-2ymqxhtffqc3jckqhwc499.streamlit.app/

## 📌 Project Overview

The **Financial Fraud Detection System** is a Machine Learning-based application developed to identify potentially fraudulent financial transactions through transaction data analysis and classification models.

The project uses a financial fraud dataset containing 5,000 transactions. It applies data preprocessing, exploratory data analysis, feature analysis, and multiple Machine Learning algorithms to identify suspicious transactions.

An interactive Streamlit dashboard allows users to explore datasets, analyze fraud patterns, compare model performance, adjust classification thresholds, and predict whether a transaction is potentially fraudulent.

## 🎯 Project Objectives

- Identify potentially fraudulent financial transactions.
- Analyze transaction patterns and fraud distribution.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Visualize fraud-related patterns and relationships.
- Train and compare Machine Learning classification models.
- Evaluate models using classification metrics.
- Analyze and adjust fraud detection thresholds.
- Build an interactive Streamlit dashboard.
- Generate useful insights for fraud analysis.

## 📊 Dataset Information

The project uses one financial fraud detection dataset.

| Dataset Details | Value |
|---|---:|
| Total Datasets | 1 |
| Total Transactions | 5,000 |
| Total Columns | 14 |
| Fraudulent Transactions | 482 |
| Legitimate Transactions | 4,518 |
| Fraud Rate | 9.64% |

### Fraud Distribution

- **Legitimate Transactions:** 4,518 (90.36%)
- **Fraudulent Transactions:** 482 (9.64%)

## ⚙️ Project Workflow

The project follows these Machine Learning and data analysis steps:

1. Dataset Loading and Inspection
2. Dataset Analysis
3. Exploratory Data Analysis (EDA)
4. Data Cleaning and Preprocessing
5. Feature Analysis
6. Correlation Analysis
7. Machine Learning Model Training
8. Model Evaluation
9. Model Comparison
10. Decision Threshold Analysis
11. Fraud Prediction
12. Interactive Dashboard Development
13. Streamlit Deployment

## 🤖 Machine Learning Models

Three classification algorithms were explored:

### 1. Logistic Regression
A classification algorithm used to estimate the probability of fraudulent transactions.

### 2. Decision Tree
A tree-based classification algorithm that learns decision rules from transaction features.

### 3. Random Forest
An ensemble learning algorithm that combines multiple decision trees to make classification predictions.

**Preliminary Selected Model:** Logistic Regression, based on PR-AUC evaluation.

## 📈 Model Evaluation

The project includes model evaluation and comparison using:

- Accuracy
- Precision
- Recall
- F1 Score
- PR-AUC
- Confusion Matrix

Additional threshold analysis is performed to understand how changing the classification threshold affects fraud detection performance.

Evaluation outputs are available in:

- `model_metrics.json`
- `model_comparison.csv`
- `evaluation_summary.txt`
- `confusion_matrix.png`
- `threshold_analysis_results.csv`

## 🖥️ Streamlit Dashboard Features

The project includes an interactive dashboard with the following features:

### 📁 1. Dataset Overview & Analysis
- Dataset information and preview.
- Transaction count and column details.
- Fraud and legitimate transaction distribution.
- Dataset exploration.

### 📊 2. Fraud Analytics Dashboard
- Fraud distribution analysis.
- Transaction amount visualization.
- Fraud pattern analysis.
- Merchant-based fraud visualization.
- Device-based fraud visualization.
- Payment method analysis.
- Correlation heatmap.

### 🤖 3. Machine Learning Model Comparison
- Logistic Regression evaluation.
- Decision Tree evaluation.
- Random Forest evaluation.
- Performance metric comparison.
- Confusion matrix visualization.

### 🚨 4. Fraud Transaction Prediction
- Transaction feature input.
- Fraud probability prediction.
- FRAUD or LEGITIMATE classification.
- Prediction results using the saved model.

### ⚙️ 5. Adjustable Fraud Detection Threshold
- Adjustable classification threshold.
- Threshold-based prediction.
- Analysis of threshold impact on model performance.
- Fraud detection decision comparison.

### 📈 6. Interactive Visualizations
- Fraud distribution chart.
- Transaction amount distribution.
- Fraud by merchant.
- Fraud by device.
- Fraud by payment method.
- Fraud correlation heatmap.
- Model evaluation charts.

## 🗂️ Project Structure

```text
Financial-Fraud-Detection-System/
│
├── app.py
├── check_dataset.py
├── data_preprocessing.py
├── dataset_analysis.py
├── eda_dashboard.py
├── evaluate_model.py
├── train_model.py
├── threshold_analysis.py
│
├── fraud_detection_dataset.csv
├── processed_fraud_dataset.csv
├── fraud_detection_model.joblib
│
├── model_metrics.json
├── model_comparison.csv
├── evaluation_summary.txt
├── threshold_analysis_results.csv
│
├── confusion_matrix.png
├── fraud_by_device.png
├── fraud_by_merchant.png
├── fraud_by_payment_method.png
├── fraud_correlation_heatmap.png
├── fraud_distribution.png
├── transaction_amount_distribution.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine Learning |
| Plotly | Interactive visualizations |
| Streamlit | Dashboard development |
| Joblib | Model serialization |
| Git | Version control |
| GitHub | Source code hosting |

## 🚀 Live Application

Explore the deployed Financial Fraud Detection System:

<p align="center">
  <a href="https://financial-fraud-detection-system-2ymqxhtffqc3jckqhwc499.streamlit.app/">
    <strong>🚀 Open Live Streamlit Dashboard</strong>
  </a>
</p>

## 💻 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Milan-coder91/Financial-Fraud-Detection-System.git
```

### 2. Navigate to the Project Folder

```bash
cd Financial-Fraud-Detection-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## 📁 Project Outputs

The project includes the following generated outputs and artifacts:

- Processed fraud detection dataset.
- Trained Machine Learning model.
- Model evaluation metrics.
- Model comparison results.
- Threshold analysis results.
- Confusion matrix.
- Fraud distribution charts.
- Merchant-based fraud analysis.
- Device-based fraud analysis.
- Payment method analysis.
- Correlation heatmap.
- Transaction amount distribution.

## 📌 Project Status

<badge color="success">Initial Development Completed</badge>

The following work has been completed:

- Dataset inspection and analysis.
- Data preprocessing.
- Exploratory Data Analysis.
- Feature and correlation analysis.
- Logistic Regression, Decision Tree, and Random Forest model evaluation.
- Model comparison.
- Threshold analysis.
- Fraud prediction functionality.
- Interactive Streamlit dashboard.
- GitHub repository setup.
- Live Streamlit deployment.

Further model validation and performance improvements are recommended before production use.

## ⚠️ Disclaimer

This project is developed for educational, learning, and demonstration purposes.

Machine Learning predictions are not guaranteed to be correct. The model may produce false-positive and false-negative predictions.

This application should not be used as the sole basis for real banking, financial, or security decisions.

## 👨‍💻 Author

**Milan Kumar**

Data Analysis | Data Science | Machine Learning | Python

---

⭐ If you find this project interesting, consider giving the GitHub repository a star!

**Project Name:** Financial Fraud Detection System

**Category:** Machine Learning | Data Science | Financial Analytics

**Application:** Financial Fraud Detection & Transaction Analysis

**GitHub:** https://github.com/Milan-coder91/Financial-Fraud-Detection-System

**Live Demo:** https://financial-fraud-detection-system-2ymqxhtffqc3jckqhwc499.streamlit.app/
