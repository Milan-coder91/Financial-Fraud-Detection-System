
# ============================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# STEP 19: FRAUD PREDICTION THRESHOLD ANALYSIS
# ============================================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. LOAD THE DATASET AND TRAINED MODEL
# ------------------------------------------------------------

df = pd.read_csv("processed_fraud_dataset.csv")
model = joblib.load("fraud_detection_model.joblib")

df = df.dropna(subset=["Fraudulent"]).copy()
df["Fraudulent"] = df["Fraudulent"].astype(int)

excluded_columns = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Fraudulent"
]

X = df.drop(columns=excluded_columns, errors="ignore")
y = df["Fraudulent"]

# ------------------------------------------------------------
# 2. CREATE THE SAME TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# 3. CALCULATE FRAUD PROBABILITIES
# ------------------------------------------------------------

fraud_probabilities = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 4. COMPARE DIFFERENT CLASSIFICATION THRESHOLDS
# ------------------------------------------------------------

thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
results = []

for threshold in thresholds:
    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "True Negatives": tn
    })

# ------------------------------------------------------------
# 5. DISPLAY AND SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== THRESHOLD ANALYSIS ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "threshold_analysis_results.csv",
    index=False
)

print("\nAnalysis completed successfully.")
print("Saved: threshold_analysis_results.csv")
