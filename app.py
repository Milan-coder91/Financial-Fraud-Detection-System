
# ============================================================
# FINANCIAL FRAUD DETECTION SYSTEM
# STEP 20: STREAMLIT FRAUD ANALYTICS DASHBOARD
# ============================================================

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed_fraud_dataset.csv"
MODEL_PATH = BASE_DIR / "fraud_detection_model.joblib"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"

# ------------------------------------------------------------
# 2. LOAD PROJECT FILES
# ------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_comparison():
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
comparison = load_comparison()

# ------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION
# ------------------------------------------------------------

if "fraud_probability" not in st.session_state:
    st.session_state.fraud_probability = None

if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None

# ------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.title("🛡️ FraudShield AI")
st.sidebar.caption("Financial Fraud Detection System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "Fraud Analytics",
        "Model Comparison",
        "Fraud Prediction"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Analytics based on the supplied transaction dataset."
)

# ------------------------------------------------------------
# 5. HOME PAGE
# ------------------------------------------------------------

if page == "Home":

    st.title("🛡️ FraudShield AI")
    st.subheader(
        "Financial Fraud Detection & Analytics Platform"
    )

    st.write(
        "Explore transaction patterns, review fraud analytics, "
        "compare machine learning models, and estimate fraud risk "
        "for a transaction."
    )

    total_transactions = len(df)
    fraud_count = int(df["Fraudulent"].sum())
    legitimate_count = total_transactions - fraud_count

    fraud_rate = (
        fraud_count / total_transactions * 100
        if total_transactions else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total_transactions:,}")
    c2.metric("Fraud Cases", f"{fraud_count:,}")
    c3.metric("Legitimate Cases", f"{legitimate_count:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.markdown("### Project Highlights")

    st.write("- Transaction dataset exploration")
    st.write("- Fraud patterns by merchant and payment method")
    st.write("- Machine learning model comparison")
    st.write("- Interactive transaction risk prediction")

    st.caption(
        "Predictions are model estimates and should not be treated "
        "as confirmed fraud findings."
    )

# ------------------------------------------------------------
# 6. DATASET OVERVIEW
# ------------------------------------------------------------

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.markdown("### Column Information")

    schema = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        schema,
        use_container_width=True
    )

    st.markdown("### Statistical Summary")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

# ------------------------------------------------------------
# 7. FRAUD ANALYTICS
# ------------------------------------------------------------

elif page == "Fraud Analytics":

    st.title("📈 Fraud Analytics Dashboard")

    fraud_labels = df["Fraudulent"].map({
        0: "Legitimate",
        1: "Fraudulent"
    }).fillna("Unknown")

    counts = fraud_labels.value_counts().rename_axis(
        "Transaction Status"
    ).reset_index(name="Transactions")

    left, right = st.columns(2)

    with left:

        st.markdown("### Transaction Class Distribution")

        fig = px.pie(
            counts,
            names="Transaction Status",
            values="Transactions",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown("### Transaction Amount Distribution")

        if "Transaction_Amount" in df.columns:

            fig = px.histogram(
                df,
                x="Transaction_Amount",
                color=fraud_labels,
                nbins=40,
                barmode="overlay",
                opacity=0.7,
                labels={"color": "Transaction Status"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    category_options = [
        col for col in [
            "Merchant_Category",
            "Payment_Method",
            "Device_Type",
            "Location",
            "International_Status"
        ]
        if col in df.columns
    ]

    if category_options:

        selected_category = st.selectbox(
            "Analyze fraud by category",
            category_options
        )

        category_data = df.groupby(
            [selected_category, "Fraudulent"],
            dropna=False
        ).size().reset_index(name="Transactions")

        category_data["Transaction Status"] = (
            category_data["Fraudulent"].map({
                0: "Legitimate",
                1: "Fraudulent"
            }).fillna("Unknown")
        )

        fig = px.bar(
            category_data,
            x=selected_category,
            y="Transactions",
            color="Transaction Status",
            barmode="group",
            title=f"Transactions by {selected_category}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    if "Transaction_Amount" in df.columns:

        st.markdown(
            "### Transaction Amount by Fraud Status"
        )

        amount_summary = df.groupby("Fraudulent")[
            "Transaction_Amount"
        ].agg([
            "count",
            "mean",
            "median",
            "sum"
        ]).reset_index()

        amount_summary["Fraud Status"] = (
            amount_summary["Fraudulent"].map({
                0: "Legitimate",
                1: "Fraudulent"
            })
        )

        st.dataframe(
            amount_summary.drop(
                columns=["Fraudulent"]
            ).round(2),
            use_container_width=True
        )

# ------------------------------------------------------------
# 8. MODEL COMPARISON
# ------------------------------------------------------------

elif page == "Model Comparison":

    st.title("🤖 Machine Learning Model Comparison")

    if comparison.empty:

        st.warning(
            "Model comparison file is not available."
        )

    else:

        st.dataframe(
            comparison.round(4),
            use_container_width=True
        )

        metric_options = [
            metric for metric in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC"
            ]
            if metric in comparison.columns
        ]

        if metric_options and "Model" in comparison.columns:

            selected_metric = st.selectbox(
                "Select evaluation metric",
                metric_options,
                index=(
                    metric_options.index("PR-AUC")
                    if "PR-AUC" in metric_options else 0
                )
            )

            fig = px.bar(
                comparison,
                x="Model",
                y=selected_metric,
                color="Model",
                title=f"Model Comparison: {selected_metric}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    if METRICS_PATH.exists():

        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metrics_info = json.load(file)

        st.caption(
            "Preliminary selected model: "
            + metrics_info.get(
                "preliminary_best_model",
                "Not available"
            )
        )

        st.warning(
            "These are preliminary test-set comparison results. "
            "Independent validation and threshold selection are "
            "recommended before production use."
        )

# ------------------------------------------------------------
# 9. FRAUD PREDICTION
# ------------------------------------------------------------

elif page == "Fraud Prediction":

    st.title("🔍 AI Fraud Prediction Center")

    st.write(
        "Enter transaction details to estimate the probability "
        "of fraud using the trained machine learning pipeline."
    )

    # --------------------------------------------------------
    # 9.1 TRANSACTION INPUT FORM
    # --------------------------------------------------------

    with st.form("fraud_prediction_form"):

        st.markdown("### Transaction Details")

        col1, col2 = st.columns(2)

        with col1:

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=100.0,
                step=10.0
            )

            average_spend = st.number_input(
                "Customer Average Spend",
                min_value=0.0,
                value=250.0,
                step=10.0
            )

            previous_transactions = st.number_input(
                "Previous Transactions",
                min_value=0,
                value=20,
                step=1
            )

            account_age = st.number_input(
                "Account Age (Days)",
                min_value=0,
                value=365,
                step=30
            )

        with col2:

            merchant = st.selectbox(
                "Merchant Category",
                sorted(
                    df["Merchant_Category"].dropna().unique()
                )
            )

            payment = st.selectbox(
                "Payment Method",
                sorted(
                    df["Payment_Method"].dropna().unique()
                )
            )

            device = st.selectbox(
                "Device Type",
                sorted(
                    df["Device_Type"].dropna().unique()
                )
            )

            location = st.selectbox(
                "Location",
                sorted(
                    df["Location"].dropna().unique()
                )
            )

            international = st.selectbox(
                "International Transaction",
                [0, 1],
                format_func=lambda x:
                    "Yes" if x == 1 else "No"
            )

            suspicious = st.selectbox(
                "Suspicious Keyword",
                sorted(
                    df["Suspicious_Keyword"].dropna().unique()
                )
            )

        st.markdown("### Transaction Time")

        time_col1, time_col2, time_col3, time_col4 = (
            st.columns(4)
        )

        with time_col1:
            hour = st.slider("Hour", 0, 23, 12)

        with time_col2:
            day = st.slider("Day", 1, 31, 15)

        with time_col3:
            month = st.slider("Month", 1, 12, 6)

        with time_col4:
            weekday = st.slider("Weekday", 0, 6, 2)

        submitted = st.form_submit_button(
            "Analyze Transaction",
            type="primary"
        )

    # --------------------------------------------------------
    # 9.2 GENERATE FRAUD PREDICTION
    # --------------------------------------------------------

    if submitted:

        input_row = pd.DataFrame([{
            "Transaction_Amount": amount,
            "Merchant_Category": merchant,
            "Payment_Method": payment,
            "Device_Type": device,
            "Location": location,
            "Is_International": international,
            "Previous_Transactions": previous_transactions,
            "Average_Spend": average_spend,
            "Account_Age_Days": account_age,
            "Suspicious_Keyword": suspicious,
            "Transaction_Hour": hour,
            "Transaction_Day": day,
            "Transaction_Month": month,
            "Transaction_Weekday": weekday,
            "Amount_to_Average_Ratio": (
                amount / average_spend
                if average_spend else 0
            ),
            "Above_Average_Spend": int(
                amount > average_spend
            ),
            "International_Status": (
                "International"
                if international else "Domestic"
            )
        }])

        try:

            model = load_model()

            probability = float(
                model.predict_proba(input_row)[0, 1]
            )

            st.session_state.fraud_probability = probability
            st.session_state.prediction_error = None

        except Exception as error:

            st.session_state.fraud_probability = None
            st.session_state.prediction_error = str(error)

    # --------------------------------------------------------
    # 9.3 ADJUSTABLE FRAUD THRESHOLD
    # --------------------------------------------------------

    st.markdown("### Fraud Alert Settings")

    threshold = st.slider(
        "Fraud Alert Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        key="fraud_threshold"
    )

    # --------------------------------------------------------
    # 9.4 DISPLAY PERSISTENT PREDICTION RESULT
    # --------------------------------------------------------

    if st.session_state.prediction_error:

        st.error(
            "Prediction could not be completed: "
            + st.session_state.prediction_error
        )

    elif st.session_state.fraud_probability is not None:

        probability = st.session_state.fraud_probability

        st.metric(
            "Estimated Fraud Probability",
            f"{probability:.1%}"
        )

        if probability >= threshold:

            st.error(
                "⚠️ Flagged for fraud review"
            )

        else:

            st.success(
                "No fraud alert at the selected threshold"
            )

        st.caption(
            "This is a model estimate, not a confirmed fraud verdict. "
            "Thresholds should be validated before real-world use."
        )

    else:

        st.info(
            "Enter transaction details and click "
            "'Analyze Transaction' to generate a prediction."
        )

# ------------------------------------------------------------
# 10. FOOTER
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "FraudShield AI | Educational project"
)
