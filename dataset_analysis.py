
# ============================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# STEP 6: DATASET ANALYSIS AND FRAUD DISTRIBUTION
# ============================================================

import pandas as pd

# Load the financial fraud detection dataset

df = pd.read_csv("fraud_detection_dataset.csv")

# Display the dataset dimensions

print("\n========== DATASET OVERVIEW ==========")
print("Total Transactions:", df.shape[0])
print("Total Columns:", df.shape[1])

# Display the data types of all columns

print("\n========== COLUMN DATA TYPES ==========")
print(df.dtypes)

# Display the distribution of fraudulent and legitimate transactions

print("\n========== FRAUD DISTRIBUTION ==========")
fraud_counts = df["Fraudulent"].value_counts().sort_index()

print("0 = Legitimate Transaction")
print("1 = Fraudulent Transaction")
print(fraud_counts)

# Calculate fraud percentages

print("\n========== FRAUD PERCENTAGES ==========")
fraud_percentages = (
    df["Fraudulent"].value_counts(normalize=True).sort_index() * 100
)

print(fraud_percentages.round(2))

# Display descriptive statistics for numerical columns

print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe().round(2).to_string())

# Display unique values in categorical columns

print("\n========== CATEGORICAL SUMMARY ==========")
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].nunique(), "unique values")
    print(df[column].value_counts().head(5).to_string())

print("\n========== ANALYSIS COMPLETED ==========")
