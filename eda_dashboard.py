
# ============================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# STEP 10: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the financial fraud detection dataset
df = pd.read_csv("fraud_detection_dataset.csv")

# Configure the chart appearance
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Convert the transaction date into datetime format
df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# ------------------------------------------------------------
# CHART 1: Fraudulent vs Legitimate Transactions
# ------------------------------------------------------------
print("Creating Chart 1: Fraud Distribution")

fraud_counts = df["Fraudulent"].map({
    0: "Legitimate",
    1: "Fraudulent"
}).value_counts()

plt.figure()
sns.barplot(x=fraud_counts.index, y=fraud_counts.values)
plt.title("Fraudulent vs Legitimate Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("fraud_distribution.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 2: Transaction Amount Distribution
# ------------------------------------------------------------
print("Creating Chart 2: Transaction Amount Distribution")

plt.figure()
sns.histplot(
    data=df,
    x="Transaction_Amount",
    hue="Fraudulent",
    bins=30,
    kde=True,
    element="step"
)
plt.title("Transaction Amount Distribution by Fraud Status")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("transaction_amount_distribution.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 3: Fraud Rate by Merchant Category
# ------------------------------------------------------------
print("Creating Chart 3: Fraud Rate by Merchant Category")

merchant_fraud_rate = (
    df.groupby("Merchant_Category")["Fraudulent"].mean() * 100
).sort_values(ascending=False)

plt.figure()
sns.barplot(
    x=merchant_fraud_rate.values,
    y=merchant_fraud_rate.index
)
plt.title("Fraud Rate by Merchant Category")
plt.xlabel("Fraud Rate (%)")
plt.ylabel("Merchant Category")
plt.tight_layout()
plt.savefig("fraud_by_merchant.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 4: Fraud Rate by Payment Method
# ------------------------------------------------------------
print("Creating Chart 4: Fraud Rate by Payment Method")

payment_fraud_rate = (
    df.groupby("Payment_Method")["Fraudulent"].mean() * 100
).sort_values(ascending=False)

plt.figure()
sns.barplot(
    x=payment_fraud_rate.values,
    y=payment_fraud_rate.index
)
plt.title("Fraud Rate by Payment Method")
plt.xlabel("Fraud Rate (%)")
plt.ylabel("Payment Method")
plt.tight_layout()
plt.savefig("fraud_by_payment_method.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 5: Fraud Rate by Device Type
# ------------------------------------------------------------
print("Creating Chart 5: Fraud Rate by Device Type")

device_fraud_rate = (
    df.groupby("Device_Type")["Fraudulent"].mean() * 100
).sort_values(ascending=False)

plt.figure()
sns.barplot(
    x=device_fraud_rate.values,
    y=device_fraud_rate.index
)
plt.title("Fraud Rate by Device Type")
plt.xlabel("Fraud Rate (%)")
plt.ylabel("Device Type")
plt.tight_layout()
plt.savefig("fraud_by_device.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 6: Correlation Heatmap for Numerical Features
# ------------------------------------------------------------
print("Creating Chart 6: Correlation Heatmap")

numerical_columns = [
    "Transaction_Amount",
    "Is_International",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
    "Fraudulent"
]

correlation_matrix = df[numerical_columns].corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("fraud_correlation_heatmap.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# FINAL SUMMARY: Confirm Chart Generation
# ------------------------------------------------------------
print("\n========== EDA COMPLETED ==========")
print("Total Transactions:", len(df))
print("Total Fraud Cases:", int(df["Fraudulent"].sum()))
print("Total Legitimate Cases:", int((df["Fraudulent"] == 0).sum()))
print("\nGenerated Chart Files:")
print("1. fraud_distribution.png")
print("2. transaction_amount_distribution.png")
print("3. fraud_by_merchant.png")
print("4. fraud_by_payment_method.png")
print("5. fraud_by_device.png")
print("6. fraud_correlation_heatmap.png")
print("\nAll EDA charts have been saved successfully.")
