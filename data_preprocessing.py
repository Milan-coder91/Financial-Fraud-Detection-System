
# ============================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# STEP 12: DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================

import pandas as pd

# Load the original financial fraud dataset
df = pd.read_csv("fraud_detection_dataset.csv")

# Convert transaction dates into datetime format
df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# Extract useful date and time features
df["Transaction_Hour"] = df["Transaction_Date"].dt.hour
df["Transaction_Day"] = df["Transaction_Date"].dt.day
df["Transaction_Month"] = df["Transaction_Date"].dt.month
df["Transaction_Weekday"] = df["Transaction_Date"].dt.dayofweek

# Create a transaction amount compared with customer average spend
df["Amount_to_Average_Ratio"] = (
    df["Transaction_Amount"]
    / df["Average_Spend"].replace(0, pd.NA)
)

# Mark transactions that exceed the customer's average spend
df["Above_Average_Spend"] = (
    df["Transaction_Amount"] > df["Average_Spend"]
).astype(int)

# Convert the international transaction indicator into readable categories
df["International_Status"] = df["Is_International"].map({
    0: "Domestic",
    1: "International"
})

# Remove rows with invalid transaction dates, if any
df = df.dropna(subset=["Transaction_Date"]).copy()

# Save the processed dataset without modifying the original CSV
df.to_csv("processed_fraud_dataset.csv", index=False)

# Display preprocessing results
print("\n========== DATA PREPROCESSING SUMMARY ==========")
print("Original dataset file: fraud_detection_dataset.csv")
print("Processed dataset file: processed_fraud_dataset.csv")
print("Processed rows:", len(df))
print("Processed columns:", len(df.columns))
print("\nNew engineered features:")
print(
    "Transaction_Hour",
    "Transaction_Day",
    "Transaction_Month",
    "Transaction_Weekday",
    "Amount_to_Average_Ratio",
    "Above_Average_Spend",
    "International_Status",
    sep="\n"
)

print("\nMissing values after date processing:", df.isnull().sum().sum())
print("\nPreprocessing completed successfully.")
