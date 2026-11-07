import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from io import BytesIO
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Financial Fraud Detection & Risk Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "fraud_detection_dataset.csv"
PROCESSED_PATH = BASE_DIR / "processed_fraud_dataset.csv"
MODEL_PATH = BASE_DIR / "fraud_detection_model.joblib"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"

TARGET = "Fraudulent"

        pd.read_csv(PROCESSED_PATH)
        if PROCESSED_PATH.exists()
        else raw.copy()
    )

    return raw, processed


try:
    df, processed = load_data()

except Exception as exc:
    st.error(f"Dataset loading error: {exc}")
    st.info(
        "fraud_detection_dataset.csv ko app.py ke same folder mein rakho."
    )
    st.stop()


if TARGET not in df.columns:
    st.error(
        f"Required target column '{TARGET}' dataset mein nahi mila."
    )
    st.stop()


df = df.copy()

df[TARGET] = (
    pd.to_numeric(df[TARGET], errors="coerce")
    .fillna(0)
    .astype(int)
    .clip(0, 1)
)


# =========================================================
# 4. COLUMN DETECTION
# =========================================================

AMOUNT_COL = next(
    (
        c for c in [
            "Transaction_Amount",
            "Amount",
            "TransactionAmount"
        ]
        if c in df.columns
    ),
    None
)

DATE_COL = next(
    (
        c for c in [
            "Transaction_Date",
            "TransactionDate",
            "Date",
            "date"
        ]
        if c in df.columns
    ),
    None
)

if AMOUNT_COL:
    df[AMOUNT_COL] = pd.to_numeric(
        df[AMOUNT_COL],
        errors="coerce"
    )

if DATE_COL:
    df["_parsed_date"] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
        dayfirst=True
    )
else:
    df["_parsed_date"] = pd.NaT

    "💰 Transaction Amount": "Transaction Amount",
    "🏪 Merchant Analysis": "Merchant Analysis",
    "💳 Transaction Type": "Transaction Type",
    "⏰ Time Based Fraud": "Time Based Fraud",
    "🔎 Detailed Investigation": "Detailed Investigation",
    "🤖 Fraud Prediction": "Fraud Prediction",
    "📁 Dataset Overview": "Dataset Overview",
    "📉 EDA Dashboard": "EDA Dashboard",
    "⚙️ Feature Engineering": "Feature Engineering",
    "🕵️ Fraud Intelligence": "Fraud Intelligence",
    "🏆 Model Comparison": "Model Comparison",
    "🧠 Explainable AI": "Explainable AI",
    "📡 Real-Time Monitoring": "Real-Time Monitoring",
    "🔮 Fraud Forecasting": "Fraud Forecasting",
    "🔗 Graph Analytics": "Graph Analytics",
    "📤 Downloads & Reports": "Downloads & Reports",
    "💬 AI Chat Assistant": "AI Chat Assistant"
}

selected_page = st.sidebar.radio(
    "NAVIGATION",
    list(pages.keys()),
    index=0,
    label_visibility="collapsed"
)

page = pages[selected_page]

st.sidebar.divider()

st.sidebar.markdown("### 🔎 Fraud Filters")

filtered = filtered_frame()

st.sidebar.caption(
    f"Showing {len(filtered):,} of {total:,} transactions"
)

st.sidebar.divider()

st.sidebar.caption("Financial Fraud Detection & Risk Analytics")


# =========================================================
# GLOBAL APPLICATION HEADER
# =========================================================

st.markdown("""
<div class="top-brand">
    <h1>💳 Financial Fraud Detection &amp; Risk Analytics</h1>
    <p>AMDOX TECHNOLOGIES | INTERNSHIP PROJECT</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 8. EXECUTIVE DASHBOARD
# =========================================================

if page == "Executive Dashboard":

    st.subheader("📊 Executive Overview")

    kpi_row(filtered)

    st.write("")

    fraud_filtered = filtered[filtered[TARGET] == 1]

    extra = st.columns(4)

    extra[0].metric(
        "✅ Legitimate Transactions",
        f"{len(filtered) - len(fraud_filtered):,}"
    )

    extra[1].metric(
        "💳 Average Amount",
        f"${filtered[AMOUNT_COL].mean():,.2f}"
        if AMOUNT_COL and len(filtered) else "N/A"
    )

    extra[2].metric(
        "🚨 Fraud Amount",
        f"${fraud_filtered[AMOUNT_COL].sum():,.2f}"
        if AMOUNT_COL else "N/A"
    )

    extra[3].metric(
        "📋 Duplicate Rows",
        f"{int(filtered.duplicated().sum()):,}"
    )

    st.divider()
            "Device_Type"
        ]
        if c in filtered.columns
    ]

    if candidates:

        selected = st.selectbox(
            "Choose Transaction Category",
            candidates
        )

        category_chart(
            filtered,
            selected,
            f"Fraud by {selected}",
            25
        )

    else:
        st.info("No transaction category columns found.")


# =========================================================
# 15. TIME BASED FRAUD
# =========================================================

elif page == "Time Based Fraud":

    page_header(
        "⏰ Time-Based Fraud Analysis",
        "Explore fraud activity by hour, weekday, and month"
    )

    if DATE_COL and filtered["_parsed_date"].notna().any():

        time_df = filtered.dropna(
            subset=["_parsed_date"]
        ).copy()

        time_df["Hour"] = time_df["_parsed_date"].dt.hour
        time_df["Weekday"] = time_df["_parsed_date"].dt.day_name()
        time_df["Month"] = time_df["_parsed_date"].dt.month_name()

        c1, c2 = st.columns(2)

        with c1:

            hourly = (
                time_df.groupby("Hour")[TARGET]
                .sum()
                .reset_index(name="Fraud Cases")
            )

            fig = px.bar(
                hourly,
                x="Hour",
                y="Fraud Cases",
                title="Fraud Cases by Hour"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        with c2:

            weekday_order = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ]

            weekday = (
                time_df.groupby("Weekday")[TARGET]
                .sum()
                .reindex(weekday_order)
                .reset_index(name="Fraud Cases")
            )

            fig = px.bar(
                weekday,
                x="Weekday",
                y="Fraud Cases",
                title="Fraud Cases by Weekday"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        monthly = (
            time_df.groupby("Month")[TARGET]
            .sum()
            .reset_index(name="Fraud Cases")
        )
        if AMOUNT_COL:

            fig = px.histogram(
                filtered,
                x=AMOUNT_COL,
                color=TARGET,
                nbins=40,
                barmode="overlay",
                title="Transaction Amount Distribution",
                color_discrete_map={0: BLUE, 1: RED}
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

    for col in [
        "Merchant_Category",
        "Payment_Method",
        "Device_Type",
        "Location"
    ]:

        if col in filtered.columns:

            category_chart(
                filtered,
                col,
                f"Fraud by {col}"
            )

    numeric = filtered.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] > 1:

        corr_data = numeric.sample(
            min(len(numeric), 50000),
            random_state=42
        ) if len(numeric) > 50000 else numeric

        fig = px.imshow(
            corr_data.corr(),
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Numeric Feature Correlation"
        )

        st.plotly_chart(
            chart_style(fig, 600),
            use_container_width=True
        )


# =========================================================
# 20. FEATURE ENGINEERING
# =========================================================

elif page == "Feature Engineering":

    page_header(
        "⚙️ Feature Engineering",
        "Inspect engineered features in the processed dataset"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Processed Rows",
        f"{len(processed):,}"
    )

    c2.metric(
        "Processed Columns",
        f"{processed.shape[1]:,}"
    )
            )

        if "merchant" in q and "Merchant_Category" in data.columns:

            return (
                "Merchant categories: "
                + ", ".join(
                    data["Merchant_Category"]
                    .dropna()
                    .astype(str)
                    .unique()[:15]
                )
            )

        if "payment" in q and "Payment_Method" in data.columns:

            return (
                "Payment methods: "
                + ", ".join(
                    data["Payment_Method"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )

        if "help" in q or "what can you do" in q:

            return (
                "You can ask about total transactions, fraud cases, "
                "fraud rate, legitimate transactions, average amount, "
                "maximum amount, missing values, duplicate rows, "
                "dataset columns, location, merchants and payment methods."
            )

        return (
            "Sorry, I could not understand that question. "
            "Try: 'Show fraud rate' or 'How many fraud cases?'"
        )

    for message in st.session_state.dashboard_chat_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    question = st.chat_input(
        "Ask about fraud, transactions, amount, location..."
    )

    if question:

        st.session_state.dashboard_chat_history.append({
            "role": "user",
            "content": question
        })

        answer = chatbot_answer(
            question,
            filtered
        )

        st.session_state.dashboard_chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()

    if st.button("🗑️ Clear Chat"):

        st.session_state.dashboard_chat_history = []

        st.rerun()

    st.divider()

    st.subheader("💡 Try These Questions")

    examples = [
        "Show total transactions",
        "How many fraud cases?",
        "What is the fraud rate?",
        "How many legitimate transactions?",
        "What is the average amount?",
        "What is the maximum amount?",
        "Show missing values",
        "How many duplicate rows?",
        "Show dataset columns",
        "Show available locations",
        "Show merchant categories",
        "Show payment methods"
    ]

    for example in examples:

        st.markdown(f"- `{example}`")


# =========================================================
# 29. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#8b7b6d;
        font-size:12px;
        padding:10px;
    ">
        💳 <b>Financial Fraud Detection &amp; Risk Analytics</b> |
        Amdox Technologies Internship Project |
        Machine Learning & Analytics
    </div>
    """,
    unsafe_allow_html=True
)








