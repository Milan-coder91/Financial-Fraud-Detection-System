
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FraudShield AI | Enterprise Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "fraud_detection_dataset.csv"
PROCESSED_PATH = BASE_DIR / "processed_fraud_dataset.csv"
MODEL_PATH = BASE_DIR / "fraud_detection_model.joblib"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"

# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e8eef8;
}
[data-testid="stSidebar"] {
    background: #111d32;
    border-right: 1px solid #263b55;
}
[data-testid="stMetric"] {
    background: #172942;
    border: 1px solid #29415f;
    padding: 18px;
    border-radius: 12px;
}
[data-testid="stMetricLabel"] {
    color: #a9bdd7;
}
[data-testid="stMetricValue"] {
    color: #ffffff;
}
h1, h2, h3 {
    color: #f4f7fc !important;
}
p, label {
    color: #c5d2e3;
}
.hero {
    background: linear-gradient(115deg, #123b75, #1769bd, #173d83);
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 25px;
    border: 1px solid #3d75bd;
}
.hero h1 {
    color: white !important;
    font-size: 38px;
}
.hero p {
    color: #e2edff;
    font-size: 16px;
}
.section-note {
    color: #9eb3cd;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA LOADING
# =====================================================

@st.cache_data
def load_data():
    raw = pd.read_csv(DATA_PATH)
    processed = (
        pd.read_csv(PROCESSED_PATH)
        if PROCESSED_PATH.exists()
        else raw.copy()
    )
    return raw, processed

try:
    df, processed = load_data()
except Exception as error:
    st.error(f"Dataset loading error: {error}")
    st.stop()

if "Fraudulent" not in df.columns:
    st.error("Fraudulent target column is missing.")
    st.stop()

df["Fraudulent"] = pd.to_numeric(
    df["Fraudulent"], errors="coerce"
).fillna(0).astype(int)

total = len(df)
fraud_count = int(df["Fraudulent"].sum())
legitimate_count = total - fraud_count
fraud_rate = fraud_count / total * 100 if total else 0

fraud_df = df[df["Fraudulent"] == 1].copy()
legitimate_df = df[df["Fraudulent"] == 0].copy()

# =====================================================
# CHART THEME
# =====================================================

def style_chart(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111d32",
        font_color="#e8eef8",
        height=height,
        margin=dict(l=25, r=25, t=45, b=25)
    )
    return fig

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 🛡️ FraudShield AI")
st.sidebar.caption("ENTERPRISE FRAUD DETECTION")
st.sidebar.divider()

pages = [
    "Executive Dashboard",
    "Dataset Overview",
    "EDA Dashboard",
    "Feature Engineering",
    "Fraud Intelligence",
    "Model Comparison",
    "Fraud Prediction",
    "Fraud Analytics",
    "Explainable AI",
    "Real-Time Monitoring",
    "Fraud Forecasting",
    "Graph Analytics"
]

page = st.sidebar.radio("NAVIGATION", pages)

st.sidebar.divider()
st.sidebar.caption("Financial Fraud Detection Platform")
st.sidebar.caption(f"Dataset: {total:,} transactions")

# =====================================================
# 1. EXECUTIVE DASHBOARD
# =====================================================

if page == "Executive Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ FraudShield AI</h1>
        <p>Enterprise Fraud Detection & Risk Intelligence Platform</p>
        <p>Machine Learning • Risk Scoring • Fraud Investigation • Analytics</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Executive Overview")

    a, b, c, d = st.columns(4)
    a.metric("Total Transactions", f"{total:,}")
    b.metric("Fraud Cases", f"{fraud_count:,}")
    c.metric("Legitimate Cases", f"{legitimate_count:,}")
    d.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.divider()
    st.subheader("Project Overview")

    st.write(
        "FraudShield AI analyzes financial transaction patterns "
        "and uses machine learning to identify potentially fraudulent activity."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Fraud Distribution")
        fig = px.pie(
            names=["Legitimate", "Fraudulent"],
            values=[legitimate_count, fraud_count],
            hole=0.55,
            color_discrete_sequence=["#16a085", "#ef4444"]
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        st.subheader("Transaction Amount Distribution")
        fig = px.histogram(
            df,
            x="Transaction_Amount",
            color="Fraudulent",
            nbins=35,
            color_discrete_sequence=["#2785e8", "#ef4444"]
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

# =====================================================
# 2. DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.title("📁 Dataset Overview")
    st.caption("Financial Fraud Detection Dataset")

    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{df.shape[0]:,}")
    b.metric("Columns", df.shape[1])
    c.metric("Fraud Cases", f"{fraud_count:,}")
    d.metric("Non-Fraud Cases", f"{legitimate_count:,}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Dataset Quality")

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })
    st.dataframe(quality, use_container_width=True)

    st.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# =====================================================
# 3. EXPLORATORY DATA ANALYSIS
# =====================================================

elif page == "EDA Dashboard":

    st.title("📊 Exploratory Data Analysis")
    st.caption("Fraud Pattern Discovery & Behavioral Analytics")

    left, right = st.columns(2)

    with left:
        st.subheader("Fraud Class Distribution")
        fig = px.bar(
            x=["Legitimate", "Fraudulent"],
            y=[legitimate_count, fraud_count],
            labels={"x": "Transaction Class", "y": "Count"},
            color=["Legitimate", "Fraudulent"],
            color_discrete_map={
                "Legitimate": "#2785e8",
                "Fraudulent": "#ef4444"
            }
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        st.subheader("Transaction Amount")
        fig = px.histogram(
            df, x="Transaction_Amount", nbins=40,
            color="Fraudulent",
            color_discrete_sequence=["#2785e8", "#ef4444"]
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    for column, title in [
        ("Merchant_Category", "Merchant Category"),
        ("Payment_Method", "Payment Method"),
        ("Device_Type", "Device Type")
    ]:
        if column in df.columns:
            st.subheader(f"Fraud by {title}")
            summary = (
                df.groupby(column)["Fraudulent"]
                .mean().mul(100).reset_index()
            )
            summary.columns = [title, "Fraud Rate (%)"]
            fig = px.bar(
                summary, x=title, y="Fraud Rate (%)",
                color="Fraud Rate (%)",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)

    if "Customer_Age" in df.columns:
        st.subheader("Customer Age Distribution")
        fig = px.histogram(df, x="Customer_Age", color="Fraudulent")
        st.plotly_chart(style_chart(fig), use_container_width=True)

# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

elif page == "Feature Engineering":

    st.title("⚙️ Feature Engineering Dashboard")
    st.caption("Explore features created during preprocessing")

    st.metric("Processed Dataset Rows", f"{len(processed):,}")
    st.metric("Processed Dataset Features", processed.shape[1])

    engineered = [
        "Transaction_Hour",
        "Transaction_Day",
        "Transaction_Month",
        "Transaction_Weekday",
        "Amount_to_Average_Ratio",
        "Above_Average_Spend",
        "International_Status"
    ]

    available = [x for x in engineered if x in processed.columns]

    if available:
        selected = st.selectbox("Select Engineered Feature", available)

        st.subheader(f"{selected} Distribution")

        if pd.api.types.is_numeric_dtype(processed[selected]):
            fig = px.histogram(processed, x=selected, nbins=30)
            st.plotly_chart(style_chart(fig), use_container_width=True)
            st.dataframe(processed[selected].describe().to_frame())
        else:
            counts = processed[selected].value_counts().reset_index()
            counts.columns = [selected, "Count"]
            fig = px.bar(counts, x=selected, y="Count")
            st.plotly_chart(style_chart(fig), use_container_width=True)

        st.subheader("Available Engineered Features")
        st.dataframe(processed[available].head(50), use_container_width=True)
    else:
        st.warning(
            "Expected engineered feature columns were not found "
            "in the processed dataset."
        )

# =====================================================
# 5. FRAUD INTELLIGENCE CENTER
# =====================================================

elif page == "Fraud Intelligence":

    st.title("🕵️ Fraud Intelligence Center")
    st.caption("Fraud Investigation & Risk Intelligence")

    a, b, c = st.columns(3)
    a.metric("Fraud Cases", f"{fraud_count:,}")
    b.metric("International Fraud Cases", int(
        fraud_df["Is_International"].sum()
    ) if "Is_International" in df.columns else 0)
    c.metric("Average Fraud Transaction Amount",
             f"{fraud_df['Transaction_Amount'].mean():,.2f}"
             if len(fraud_df) else "N/A")

    if "Is_International" in df.columns:
        st.subheader("International Transaction Analysis")
        international = (
            df.groupby("Is_International")["Fraudulent"]
            .agg(["count", "sum", "mean"]).reset_index()
        )
        international["Fraud Rate (%)"] = international["mean"] * 100
        st.dataframe(international, use_container_width=True)

    st.subheader("Fraud by Merchant")
    if "Merchant_Category" in df.columns:
        summary = fraud_df["Merchant_Category"].value_counts().reset_index()
        summary.columns = ["Merchant Category", "Fraud Cases"]
        fig = px.bar(summary, x="Merchant Category", y="Fraud Cases",
                     color="Fraud Cases", color_continuous_scale="Reds")
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("Fraud Transaction Records")
    st.dataframe(fraud_df.head(100), use_container_width=True)

# =====================================================
# 6. MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("🏆 Model Comparison Dashboard")
    st.caption("Machine Learning Performance Evaluation")

    if COMPARISON_PATH.exists():

        results = pd.read_csv(COMPARISON_PATH)
        st.subheader("Model Performance Table")
        st.dataframe(results, use_container_width=True)

        metric_candidates = [
            "Accuracy", "Precision", "Recall", "F1",
            "ROC-AUC", "PR-AUC"
        ]

        available_metrics = [
            col for col in metric_candidates if col in results.columns
        ]

        if available_metrics:
            selected_metric = st.selectbox(
                "Select Performance Metric", available_metrics
            )

            fig = px.bar(
                results,
                x=results.columns[0],
                y=selected_metric,
                color=selected_metric,
                color_continuous_scale="Blues"
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)

        st.info(
            "Preliminary selected model: Logistic Regression, "
            "based on the PR-AUC comparison."
        )
    else:
        st.warning("Model comparison file not found.")

# =====================================================
# 7. FRAUD PREDICTION
# =====================================================

elif page == "Fraud Prediction":

    st.title("🚨 AI Fraud Prediction Center")
    st.caption("Evaluate a transaction using the trained ML model")

    if not MODEL_PATH.exists():
        st.error("Trained model file not found.")
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        st.error(f"Model loading error: {error}")
        st.stop()

    st.info(
        "Enter transaction information. The prediction is an estimate "
        "from the trained model, not a confirmed fraud investigation."
    )

    with st.form("fraud_prediction_form"):

        a, b = st.columns(2)

        with a:
            amount = st.number_input(
                "Transaction Amount", min_value=0.0, value=1000.0
            )
            merchant = st.selectbox(
                "Merchant Category", df["Merchant_Category"].dropna().unique()
            )
            payment = st.selectbox(
                "Payment Method", df["Payment_Method"].dropna().unique()
            )
            device = st.selectbox(
                "Device Type", df["Device_Type"].dropna().unique()
            )
            location = st.selectbox(
                "Location", df["Location"].dropna().unique()
            )
            international = st.selectbox(
                "International Transaction", [0, 1]
            )

        with b:
            previous = st.number_input(
                "Previous Transactions", min_value=0, value=5
            )
            average = st.number_input(
                "Average Spend", min_value=0.0, value=1000.0
            )
            age = st.number_input(
                "Account Age (Days)", min_value=0, value=365
            )
            keyword = st.selectbox(
                "Suspicious Keyword",
                df["Suspicious_Keyword"].dropna().unique()
            )
            threshold = st.slider(
                "Fraud Alert Threshold", 0.05, 0.95, 0.50, 0.05
            )

        submitted = st.form_submit_button("Analyze Transaction")

    if submitted:

        date = pd.Timestamp.now()
        ratio = amount / average if average > 0 else 0

        prediction_input = pd.DataFrame([{
            "Transaction_ID": "NEW_TRANSACTION",
            "Customer_ID": "NEW_CUSTOMER",
            "Transaction_Date": date.strftime("%d-%m-%Y %H:%M"),
            "Transaction_Amount": amount,
            "Merchant_Category": merchant,
            "Payment_Method": payment,
            "Device_Type": device,
            "Location": location,
            "Is_International": international,
            "Previous_Transactions": previous,
            "Average_Spend": average,
            "Account_Age_Days": age,
            "Suspicious_Keyword": keyword,
            "Transaction_Hour": date.hour,
            "Transaction_Day": date.day,
            "Transaction_Month": date.month,
            "Transaction_Weekday": date.weekday(),
            "Amount_to_Average_Ratio": ratio,
            "Above_Average_Spend": int(amount > average),
            "International_Status": (
                "International" if international else "Domestic"
            )
        }])

        try:
            expected = getattr(model, "feature_names_in_", None)
            if expected is not None:
                prediction_input = prediction_input.reindex(
                    columns=expected
                )

            probability = float(model.predict_proba(prediction_input)[0][1])

            st.metric("Fraud Probability", f"{probability * 100:.2f}%")

            if probability >= threshold:
                st.error("⚠️ Flagged for Fraud Review")
            else:
                st.success("✅ Below Selected Fraud Alert Threshold")

            st.caption(
                "This is a model-based screening result, not a final fraud verdict."
            )

        except Exception as error:
            st.error(f"Prediction could not be completed: {error}")
            st.info(
                "The input feature structure may need to be aligned "
                "with the saved model's training pipeline."
            )

# =====================================================
# 8. FRAUD ANALYTICS
# =====================================================

elif page == "Fraud Analytics":

    st.title("📈 Fraud Analytics Dashboard")
    st.caption("AI-Powered Financial Fraud Analysis")

    a, b, c, d = st.columns(4)
    a.metric("Total Transactions", f"{total:,}")
    b.metric("Fraud Cases", f"{fraud_count:,}")
    c.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    d.metric(
        "Average Transaction Amount",
        f"{df['Transaction_Amount'].mean():,.2f}"
    )

    st.subheader("Fraud vs Legitimate Transactions")
    fig = px.pie(
        names=["Legitimate", "Fraudulent"],
        values=[legitimate_count, fraud_count],
        hole=0.5,
        color_discrete_sequence=["#2785e8", "#ef4444"]
    )
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("Transaction Amount by Fraud Status")
    fig = px.box(
        df, x="Fraudulent", y="Transaction_Amount",
        color="Fraudulent"
    )
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("Fraud by Device Type")
    if "Device_Type" in df.columns:
        device = fraud_df["Device_Type"].value_counts().reset_index()
        device.columns = ["Device Type", "Fraud Cases"]
        fig = px.bar(device, x="Device Type", y="Fraud Cases")
        st.plotly_chart(style_chart(fig), use_container_width=True)

# =====================================================
# 9. EXPLAINABLE AI
# =====================================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI")
    st.caption("Understand the factors influencing model predictions")

    if not MODEL_PATH.exists():
        st.warning("Trained model file is not available.")
    else:
        try:
            model = joblib.load(MODEL_PATH)
            estimator = model.steps[-1][1] if hasattr(model, "steps") else model

            importance = None
            if hasattr(estimator, "feature_importances_"):
                importance = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                importance = np.abs(estimator.coef_).ravel()

            if importance is not None:
                names = None
                if hasattr(model, "steps"):
                    for _, step in model.steps[:-1]:
                        if hasattr(step, "get_feature_names_out"):
                            try:
                                names = step.get_feature_names_out()
                            except Exception:
                                pass

                if names is None or len(names) != len(importance):
                    names = [f"Feature {i+1}" for i in range(len(importance))]

                explanation = pd.DataFrame({
                    "Feature": names,
                    "Importance": importance
                }).sort_values("Importance", ascending=False).head(15)

                fig = px.bar(
                    explanation.sort_values("Importance"),
                    x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(style_chart(fig, 500), use_container_width=True)
                st.dataframe(explanation, use_container_width=True)
            else:
                st.info(
                    "This model does not expose native feature importance. "
                    "A separate permutation-importance analysis is needed."
                )

        except Exception as error:
            st.warning(f"Model explanation unavailable: {error}")

# =====================================================
# 10. REAL-TIME MONITORING
# =====================================================

elif page == "Real-Time Monitoring":

    st.title("📡 Fraud Monitoring Center")
    st.caption("Monitoring transaction records available in the dataset")

    st.info(
        "This page displays historical dataset records. "
        "It is not connected to a live banking transaction stream."
    )

    a, b, c = st.columns(3)
    a.metric("Dataset Transactions", f"{total:,}")
    b.metric("Recorded Fraud Cases", f"{fraud_count:,}")
    c.metric("Recorded Fraud Rate", f"{fraud_rate:.2f}%")

    monitor = df.copy()

    if "Transaction_Date" in monitor.columns:
        monitor["Transaction_Date"] = pd.to_datetime(
            monitor["Transaction_Date"],
            format="%d-%m-%Y %H:%M",
            errors="coerce"
        )
        monitor = monitor.sort_values(
            "Transaction_Date", ascending=False
        )

    st.subheader("Recent Transaction Records")
    st.dataframe(monitor.head(100), use_container_width=True)

    st.subheader("Recorded Fraud Transactions")
    st.dataframe(
        monitor[monitor["Fraudulent"] == 1].head(50),
        use_container_width=True
    )

# =====================================================
# 11. FRAUD FORECASTING
# =====================================================

elif page == "Fraud Forecasting":

    st.title("📉 Fraud Trend Analysis")
    st.caption("Historical fraud activity over time")

    if "Transaction_Date" not in df.columns:
        st.warning("Transaction date is not available.")
    else:
        dated = df.copy()
        dated["Transaction_Date"] = pd.to_datetime(
            dated["Transaction_Date"],
            format="%d-%m-%Y %H:%M",
            errors="coerce"
        )
        dated = dated.dropna(subset=["Transaction_Date"])

        if len(dated):
            daily = (
                dated.set_index("Transaction_Date")
                .resample("D")["Fraudulent"]
                .agg(["count", "sum"])
                .reset_index()
            )
            daily.columns = ["Date", "Transactions", "Fraud Cases"]

            st.subheader("Historical Fraud Trend")
            fig = px.line(
                daily, x="Date", y="Fraud Cases",
                markers=True
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)

            st.subheader("Daily Transaction Volume")
            fig = px.area(
                daily, x="Date", y="Transactions"
            )
            st.plotly_chart(style_chart(fig), use_container_width=True)

            st.info(
                "These are historical trends, not future predictions. "
                "A validated forecasting model is required for genuine forecasts."
            )
            st.dataframe(daily, use_container_width=True)
        else:
            st.warning("No valid transaction dates were found.")

# =====================================================
# 12. GRAPH ANALYTICS
# =====================================================

elif page == "Graph Analytics":

    st.title("🕸️ Graph Analytics Dashboard")
    st.caption("Numerical relationship discovery and correlation analysis")

    numeric = df.select_dtypes(include=np.number).copy()

    if numeric.shape[1] >= 2:

        if "Fraudulent" in numeric.columns:
            corr = numeric.corr(numeric_only=True)

            st.subheader("Correlation Heatmap")
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(style_chart(fig, 600), use_container_width=True)

            st.subheader("Correlation with Fraud Target")
            target_corr = (
                corr["Fraudulent"]
                .drop("Fraudulent")
                .sort_values()
                .reset_index()
            )
            target_corr.columns = ["Feature", "Correlation"]

            fig = px.bar(
                target_corr,
                x="Correlation", y="Feature",
                orientation="h",
                color="Correlation",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(style_chart(fig, 500), use_container_width=True)

        else:
            st.warning("Fraud target is not available for correlation.")

    else:
        st.warning("Not enough numerical columns for correlation analysis.")

# =====================================================
# FOOTER
# =====================================================

st.sidebar.divider()
st.sidebar.caption("FraudShield AI | Financial Fraud Detection")
st.sidebar.caption("Research & Analytics Dashboard")
