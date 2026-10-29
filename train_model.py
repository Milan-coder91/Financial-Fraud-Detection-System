
# ============================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# STEP 15: TRAIN AND COMPARE MACHINE LEARNING MODELS
# ============================================================

import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# 1. LOAD THE PROCESSED DATASET
# ------------------------------------------------------------
print("\n========== LOADING DATASET ==========")

df = pd.read_csv("processed_fraud_dataset.csv")

# Remove rows where the target label is missing
df = df.dropna(subset=["Fraudulent"]).copy()
df["Fraudulent"] = df["Fraudulent"].astype(int)

# ------------------------------------------------------------
# 2. PREPARE FEATURES AND TARGET
# ------------------------------------------------------------
print("\n========== PREPARING FEATURES ==========")

# Exclude identifiers and the original date string
# The extracted date/time features are retained
excluded_columns = [
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Fraudulent"
]

X = df.drop(columns=excluded_columns, errors="ignore")
y = df["Fraudulent"]

# Separate numerical and categorical features
numerical_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("Total Records:", len(df))
print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

# ------------------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------
print("\n========== SPLITTING DATA ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# ------------------------------------------------------------
# 4. BUILD THE PREPROCESSING PIPELINE
# ------------------------------------------------------------

# Impute missing values and scale numerical features
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Impute missing values and encode categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Apply the correct preprocessing to each feature type
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ------------------------------------------------------------
# 5. DEFINE THE MACHINE LEARNING MODELS
# ------------------------------------------------------------
print("\n========== INITIALIZING MODELS ==========")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
        max_depth=10
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# ------------------------------------------------------------
# 6. TRAIN MODELS AND EVALUATE PERFORMANCE
# ------------------------------------------------------------
results = []
trained_models = {}

print("\n========== MODEL TRAINING ==========")

for model_name, model in models.items():
    print(f"\nTraining: {model_name}")

    # Keep preprocessing inside the pipeline to prevent leakage
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    # Predict classes and fraud probabilities
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": average_precision_score(y_test, y_prob)
    }

    results.append(metrics)
    trained_models[model_name] = pipeline

    print(pd.Series(metrics).to_string())
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

# ------------------------------------------------------------
# 7. COMPARE MODELS AND SELECT THE BEST MODEL
# ------------------------------------------------------------
print("\n========== MODEL COMPARISON ==========")

results_df = pd.DataFrame(results).sort_values(
    by="PR-AUC",
    ascending=False
).reset_index(drop=True)

print(results_df.round(4).to_string(index=False))

# Select the model with the highest PR-AUC on the test set
# This is a preliminary comparison; final tuning should use
# validation or cross-validation before final test evaluation.
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nPreliminary Best Model:", best_model_name)

# ------------------------------------------------------------
# 8. SAVE THE TRAINED MODEL AND METRICS
# ------------------------------------------------------------
joblib.dump(best_model, "fraud_detection_model.joblib")

results_df.to_csv("model_comparison.csv", index=False)

with open("model_metrics.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "preliminary_best_model": best_model_name,
            "selection_metric": "PR-AUC",
            "test_set_is_preliminary": True,
            "metrics": results
        },
        file,
        indent=4
    )

print("\n========== TRAINING COMPLETED ==========")
print("Saved Model: fraud_detection_model.joblib")
print("Saved Comparison: model_comparison.csv")
print("Saved Metrics: model_metrics.json")
