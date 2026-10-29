
# ============================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# STEP 18: MODEL EVALUATION AND PERFORMANCE VISUALIZATION
# ============================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# ------------------------------------------------------------
# 1. LOAD DATASET AND TRAINED MODEL
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
# 3. GENERATE PREDICTIONS
# ------------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n========== MODEL EVALUATION ==========")
print("Test Records:", len(X_test))
print("Actual Fraud Cases:", int(y_test.sum()))
print("Predicted Fraud Cases:", int(y_pred.sum()))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
))

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("PR-AUC:", round(average_precision_score(y_test, y_prob), 4))

# ------------------------------------------------------------
# 4. CREATE AND SAVE CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraudulent"],
    yticklabels=["Legitimate", "Fraudulent"]
)

plt.title("Fraud Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 5. SAVE EVALUATION SUMMARY
# ------------------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraudulent"],
    zero_division=0
)

with open("evaluation_summary.txt", "w", encoding="utf-8") as file:
    file.write("FINANCIAL FRAUD DETECTION - MODEL EVALUATION\n")
    file.write(f"Test Records: {len(X_test)}\n")
    file.write(f"Actual Fraud Cases: {int(y_test.sum())}\n")
    file.write(f"Predicted Fraud Cases: {int(y_pred.sum())}\n")
    file.write(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}\n")
    file.write(
        f"PR-AUC: {average_precision_score(y_test, y_prob):.4f}\n\n"
    )
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))

print("\nEvaluation completed successfully.")
print("Saved: confusion_matrix.png")
print("Saved: evaluation_summary.txt")
