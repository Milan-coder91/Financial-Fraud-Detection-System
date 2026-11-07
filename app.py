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


# =========================================================
# 2. PROFESSIONAL UI THEME
# =========================================================

st.markdown("""
<style>
:root {
    --bg: #111315;
    --panel: #1b1e22;
    --sidebar: #17191c;
    --card: #202429;
    --accent: #4f8cff;
    --accent2: #2f6fdb;
    --text: #f3f5f7;
    --muted: #aab2bd;
    --line: #343a42;
    --danger: #ef5350;
    --success: #35b779;
}
.stApp {
    background: var(--bg);
    color: var(--text);
}
[data-testid="stHeader"] {
    background: rgba(17,19,21,0.96);
}
[data-testid="stToolbar"] { right: 1rem; }
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2.5rem;
    max-width: 1600px;
}
[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.3rem;
}
[data-testid="stSidebar"] * {
    color: var(--text);
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 5px;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent !important;
    border-radius: 9px;
    padding: 8px 10px;
    transition: all .18s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #2a2f35 !important;
}
[data-testid="stSidebar"] [role="radio"] > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
    background: #343b45 !important;
    border: 1px solid #4c5969;
    border-radius: 9px;
    font-weight: 700;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"],
[data-testid="stSidebar"] [data-testid="stDateInput"] {
    background: #22262b;
    border-radius: 9px;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #252a30 !important;
    color: var(--text) !important;
    border-color: #3b424b !important;
}
[data-testid="stSidebar"] hr { border-color: var(--line); }
.stButton button, .stDownloadButton button {
    border-radius: 9px;
    border: 1px solid #46515e;
    background: #252b32;
    color: var(--text);
    transition: .2s;
}
.stButton button[kind="primary"] {
    background: var(--accent2) !important;
    border-color: var(--accent2) !important;
    color: white !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    border-color: var(--accent);
    color: white;
}
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 17px 19px;
    box-shadow: 0 5px 18px rgba(0,0,0,.16);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; }
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 750;
}
h1, h2, h3, h4 { color: var(--text) !important; }
p, label, .stCaption { color: var(--muted); }
.section-sub {
    color: var(--muted);
    font-size: 13px;
    margin-top: -8px;
}
.top-brand {
    background: linear-gradient(115deg, #202a38 0%, #1b2736 55%, #173d68 100%);
    border: 1px solid #34485f;
    border-radius: 16px;
    padding: 23px 27px;
    margin: 0 0 25px 0;
    box-shadow: 0 8px 26px rgba(0,0,0,.2);
}
.top-brand h1 {
    color: #ffffff !important;
    font-size: clamp(24px, 3vw, 34px);
    font-weight: 800;
    margin: 0 0 7px 0;
    line-height: 1.2;
}
.top-brand p {
    color: #c8d7e8 !important;
    margin: 0;
    font-size: 13px;
    letter-spacing: .7px;
    font-weight: 650;
}
.sidebar-brand {
    background: linear-gradient(135deg, #202a38, #173d68);
    border: 1px solid #34485f;
    border-radius: 12px;
    padding: 13px 12px;
    margin-bottom: 13px;
}
.sidebar-brand h3 {
    color: white !important;
    font-size: 19px;
    margin: 0;
}
.sidebar-brand p {
    color: #c8d7e8 !important;
    font-size: 10px;
    letter-spacing: .5px;
    margin: 5px 0 0 0;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 10px;
}
[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 10px;
}
hr { border-color: var(--line); }
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. DATA LOADING
# =========================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}"
        )

    raw = pd.read_csv(DATA_PATH)

    processed = (
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


# =========================================================
# 5. GLOBAL METRICS
# =========================================================

BROWN = "#4f8cff"
ORANGE = "#f59e0b"
RED = "#ef5350"
GREEN = "#35b779"
BLUE = "#38a3ff"

total = len(df)

fraud_count = int(df[TARGET].sum())

legitimate_count = total - fraud_count

fraud_rate = (
    fraud_count / total * 100
    if total else 0
)

fraud_df = df[df[TARGET] == 1].copy()

legitimate_df = df[df[TARGET] == 0].copy()

total_amount = (
    float(df[AMOUNT_COL].fillna(0).sum())
    if AMOUNT_COL else 0
)

avg_amount = (
    float(df[AMOUNT_COL].mean())
    if AMOUNT_COL and df[AMOUNT_COL].notna().any()
    else 0
)

fraud_amount = (
    float(fraud_df[AMOUNT_COL].fillna(0).sum())
    if AMOUNT_COL else 0
)


# =========================================================
# 6. HELPER FUNCTIONS
# =========================================================

def chart_style(fig, height=390):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1b1e22",
        plot_bgcolor="#202429",
        font=dict(
            color="#f3f5f7",
            family="Arial"
        ),
        height=height,
        margin=dict(
            l=25,
            r=25,
            t=55,
            b=30
        ),
        legend_title_text=""
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#343a42"
    )

    fig.update_yaxes(
        gridcolor="#343a42",
        zerolinecolor="#343a42"
    )

    return fig


def page_header(title, subtitle):

    st.title(title)

    st.markdown(
        f'<div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True
    )

    st.divider()


def csv_bytes(frame):

    return frame.to_csv(
        index=False
    ).encode("utf-8-sig")


def kpi_row(data=None):

    data = df if data is None else data

    n = len(data)
    f = int(data[TARGET].sum()) if n else 0
    l = n - f
    rate = f / n * 100 if n else 0

    amount = (
        data[AMOUNT_COL].fillna(0).sum()
        if AMOUNT_COL else 0
    )

    cols = st.columns(4)

    cols[0].metric(
        "📊 Total Transactions",
        f"{n:,}"
    )

    cols[1].metric(
        "🚨 Fraud Transactions",
        f"{f:,}"
    )

    cols[2].metric(
        "📈 Fraud Rate",
        f"{rate:.2f}%"
    )

    cols[3].metric(
        "💰 Transaction Amount",
        f"${amount:,.0f}" if AMOUNT_COL else "N/A"
    )


def category_chart(data, column, title, top_n=15):

    if column not in data.columns:
        st.info(f"{column} column available nahi hai.")
        return

    summary = (
        data.groupby(column, dropna=False)[TARGET]
        .agg(
            Transactions="count",
            Fraud_Cases="sum",
            Fraud_Rate="mean"
        )
        .reset_index()
    )

    summary["Fraud Rate (%)"] = (
        summary["Fraud_Rate"] * 100
    )

    summary = summary.sort_values(
        "Fraud_Cases",
        ascending=False
    ).head(top_n)

    fig = px.bar(
        summary,
        x=column,
        y="Fraud_Cases",
        color="Fraud Rate (%)",
        color_continuous_scale=["#e8c9a8", "#c94f45"],
        hover_data=[
            "Transactions",
            "Fraud Rate (%)"
        ],
        title=title
    )

    st.plotly_chart(
        chart_style(fig),
        use_container_width=True
    )

    st.dataframe(
        summary.drop(columns=["Fraud_Rate"]),
        use_container_width=True
    )


def make_pdf_report(frame):

    if not REPORTLAB_AVAILABLE:
        return None

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#50382c"),
            fontSize=20,
            leading=24,
            spaceAfter=8
        )
    )

    n = len(frame)
    f = int(frame[TARGET].sum()) if n else 0
    l = n - f
    rate = f / n * 100 if n else 0

    amount = (
        float(frame[AMOUNT_COL].fillna(0).sum())
        if AMOUNT_COL else 0
    )

    story = [
        Paragraph(
            "Financial Fraud Detection & Risk Analytics",
            styles["ReportTitle"]
        ),
        Paragraph(
            "Amdox Technologies Internship Project",
            styles["Normal"]
        ),
        Spacer(1, 7 * mm)
    ]

    summary = [
        ["Metric", "Value"],
        ["Transactions", f"{n:,}"],
        ["Fraud Cases", f"{f:,}"],
        ["Legitimate Cases", f"{l:,}"],
        ["Fraud Rate", f"{rate:.2f}%"],
        [
            "Total Amount",
            f"${amount:,.2f}" if AMOUNT_COL else "Not available"
        ],
        [
            "Generated",
            datetime.now().strftime("%d %b %Y, %I:%M %p")
        ]
    ]

    table = Table(
        summary,
        colWidths=[75 * mm, 80 * mm]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#79543f")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#ded3c6")),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#fbfaf6")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#49372d")),
            ("PADDING", (0, 0), (-1, -1), 8)
        ])
    )

    story.extend([
        table,
        Spacer(1, 8 * mm),
        Paragraph(
            "Note: This report summarizes dataset records. "
            "A fraud flag is not a confirmed fraud finding.",
            styles["BodyText"]
        )
    ])

    doc.build(story)
    buffer.seek(0)

    return buffer.getvalue()


def filtered_frame():

    result = df.copy()

    if DATE_COL and result["_parsed_date"].notna().any():

        min_d = result["_parsed_date"].min().date()
        max_d = result["_parsed_date"].max().date()

        selected = st.sidebar.date_input(
            "📅 Transaction Date Range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key="date_filter"
        )

        if isinstance(selected, (tuple, list)) and len(selected) == 2:

            start, end = selected

            result = result[
                result["_parsed_date"].dt.date.between(
                    start,
                    end
                )
            ]

    if "Location" in result.columns:

        locations = sorted(
            result["Location"].dropna().astype(str).unique()
        )

        chosen = st.sidebar.multiselect(
            "📍 Location",
            locations,
            key="location_filter"
        )

        if chosen:
            result = result[
                result["Location"].astype(str).isin(chosen)
            ]

    if "Merchant_Category" in result.columns:

        merchants = sorted(
            result["Merchant_Category"].dropna().astype(str).unique()
        )

        chosen_m = st.sidebar.multiselect(
            "🏪 Merchant Category",
            merchants,
            key="merchant_filter"
        )

        if chosen_m:
            result = result[
                result["Merchant_Category"].astype(str).isin(chosen_m)
            ]

    return result


# =========================================================
# 7. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("""
<div class="sidebar-brand">
    <h3>💳 Financial Fraud</h3>
    <p>AMDOX TECHNOLOGIES | INTERNSHIP PROJECT</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

pages = {
    "🏠 Executive Dashboard": "Executive Dashboard",
    "📊 Overview": "Overview",
    "📈 Trend Analysis": "Trend Analysis",
    "📍 Location Fraud": "Location Fraud",
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
    if only_fraud:
        investigation = investigation[
            investigation[TARGET] == 1
        ]

    if high_amount and AMOUNT_COL and len(investigation):

        threshold = investigation[AMOUNT_COL].quantile(.90)

        investigation = investigation[
            investigation[AMOUNT_COL] >= threshold
        ]

    st.metric(
        "Matching Transactions",
        f"{len(investigation):,}"
    )

    st.dataframe(
        investigation.drop(
            columns=["_parsed_date"],
            errors="ignore"
        ),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Investigation Results",
        csv_bytes(
            investigation.drop(
                columns=["_parsed_date"],
                errors="ignore"
            )
        ),
        "investigation_results.csv",
        "text/csv"
    )


# =========================================================
# 17. FRAUD PREDICTION
# =========================================================

elif page == "Fraud Prediction":

    page_header(
        "🤖 Fraud Prediction",
        "Predict whether a transaction may be fraudulent using the trained model"
    )

    if MODEL_PATH.exists():

        try:
            model = joblib.load(MODEL_PATH)

            st.success("Fraud detection model loaded successfully.")

            st.subheader("Enter Transaction Details")

            feature_columns = getattr(
                model,
                "feature_names_in_",
                None
            )

            if feature_columns is None:
                feature_columns = (
                    processed.drop(
                        columns=[TARGET],
                        errors="ignore"
                    ).select_dtypes(include=np.number).columns.tolist()
                )
            else:
                feature_columns = list(feature_columns)

            input_values = {}

            columns = st.columns(2)

            for i, feature in enumerate(feature_columns):

                with columns[i % 2]:

                    if feature in df.columns:

                        if pd.api.types.is_numeric_dtype(df[feature]):

                            default_value = float(
                                df[feature].median()
                            ) if df[feature].notna().any() else 0.0

                            input_values[feature] = st.number_input(
                                feature.replace("_", " ").title(),
                                value=default_value
                            )

                        else:

                            options = (
                                df[feature]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )

                            if options:
                                input_values[feature] = st.selectbox(
                                    feature.replace("_", " ").title(),
                                    options
                                )
                            else:
                                input_values[feature] = ""

                    else:
                        input_values[feature] = st.number_input(
                            feature.replace("_", " ").title(),
                            value=0.0
                        )

            if st.button(
                "🔍 Predict Fraud",
                type="primary",
                use_container_width=True
            ):

                input_df = pd.DataFrame(
                    [input_values],
                    columns=feature_columns
                )

                prediction = model.predict(input_df)[0]

                probability = None

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_df)[0]

                    if len(probabilities) > 1:
                        probability = float(probabilities[1]) * 100

                if int(prediction) == 1:

                    st.error("🚨 Model Prediction: Potential Fraud")

                else:

                    st.success("✅ Model Prediction: Likely Legitimate")

                if probability is not None:

                    st.metric(
                        "Fraud Probability",
                        f"{probability:.2f}%"
                    )

                st.caption(
                    "This is a model prediction, not a confirmed fraud finding."
                )

        except Exception as exc:

            st.error(f"Model prediction error: {exc}")

    else:

        st.warning(
            "Trained model file not found. "
            "Place fraud_detection_model.joblib beside app.py."
        )


# =========================================================
# 18. DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":

    page_header(
        "📁 Dataset Overview",
        "Inspect dataset dimensions, columns, types, and quality"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{len(filtered):,}")
    c2.metric("Columns", f"{len(filtered.columns):,}")
    c3.metric(
        "Memory Usage",
        f"{filtered.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered.drop(
            columns=["_parsed_date"],
            errors="ignore"
        ).head(200),
        use_container_width=True
    )

    st.subheader("Column Information")

    info = pd.DataFrame({
        "Column": filtered.columns,
        "Data Type": [
            str(filtered[c].dtype)
            for c in filtered.columns
        ],
        "Missing Values": [
            int(filtered[c].isna().sum())
            for c in filtered.columns
        ],
        "Unique Values": [
            int(filtered[c].nunique(dropna=True))
            for c in filtered.columns
        ]
    })

    st.dataframe(
        info,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        filtered.describe(include="all").transpose(),
        use_container_width=True
    )


# =========================================================
# 19. EDA DASHBOARD
# =========================================================

elif page == "EDA Dashboard":

    page_header(
        "📉 Exploratory Data Analysis",
        "Explore distributions, missing values, and relationships"
    )

    st.subheader("Missing Values")

    missing = (
        filtered.isna()
        .sum()
        .reset_index()
    )

    missing.columns = ["Column", "Missing Values"]

    missing["Missing Percentage"] = (
        missing["Missing Values"] /
        max(len(filtered), 1) * 100
    )

    missing = missing.sort_values(
        "Missing Values",
        ascending=False
    )

    c1, c2 = st.columns(2)

    with c1:

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with c2:

        fig = px.bar(
            missing,
            x="Column",
            y="Missing Percentage",
            title="Missing Percentage by Column"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    st.dataframe(
        missing,
        use_container_width=True
    )

    st.subheader("Numerical Feature Distribution")

    numeric_columns = filtered.select_dtypes(
        include=np.number
    ).columns.tolist()

    numeric_columns = [
        c for c in numeric_columns
        if c != TARGET
    ]

    if numeric_columns:

        selected_numeric = st.selectbox(
            "Select Numerical Feature",
            numeric_columns
        )

        fig = px.histogram(
            filtered,
            x=selected_numeric,
            color=TARGET,
            marginal="box",
            barmode="overlay",
            title=f"Distribution of {selected_numeric}"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    st.subheader("Correlation Heatmap")

    if len(numeric_columns) >= 2:

        corr_columns = numeric_columns + [TARGET]

        corr = filtered[corr_columns].corr(
            numeric_only=True
        )

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation"
        )

        st.plotly_chart(
            chart_style(fig, 500),
            use_container_width=True
        )

    st.download_button(
        "⬇️ Download Missing Value Report",
        csv_bytes(missing),
        "missing_value_report.csv",
        "text/csv"
    )


# =========================================================
# 20. FEATURE ENGINEERING
# =========================================================

elif page == "Feature Engineering":

    page_header(
        "⚙️ Feature Engineering",
        "Create and inspect useful features for fraud analysis"
    )

    engineered = filtered.copy()

    if AMOUNT_COL:

        engineered["Amount_Log"] = np.log1p(
            engineered[AMOUNT_COL].clip(lower=0)
        )

        engineered["Amount_Band"] = pd.cut(
            engineered[AMOUNT_COL],
            bins=[
                -np.inf,
                100,
                500,
                1000,
                5000,
                np.inf
            ],
            labels=[
                "Very Low",
                "Low",
                "Medium",
                "High",
                "Very High"
            ]
        )

    if DATE_COL and engineered["_parsed_date"].notna().any():

        engineered["Transaction_Hour"] = (
            engineered["_parsed_date"].dt.hour
        )

        engineered["Transaction_Day"] = (
            engineered["_parsed_date"].dt.day
        )

        engineered["Transaction_Month"] = (
            engineered["_parsed_date"].dt.month
        )

        engineered["Transaction_Weekday"] = (
            engineered["_parsed_date"].dt.day_name()
        )

        engineered["Is_Weekend"] = (
            engineered["_parsed_date"].dt.dayofweek >= 5
        ).astype(int)

    st.subheader("Engineered Dataset Preview")

    st.dataframe(
        engineered.drop(
            columns=["_parsed_date"],
            errors="ignore"
        ).head(200),
        use_container_width=True
    )

    st.metric(
        "Total Features",
        f"{len(engineered.columns):,}"
    )

    st.download_button(
        "⬇️ Download Engineered Dataset",
        csv_bytes(
            engineered.drop(
                columns=["_parsed_date"],
                errors="ignore"
            )
        ),
        "engineered_fraud_dataset.csv",
        "text/csv"
    )


# =========================================================
# 21. FRAUD INTELLIGENCE
# =========================================================

elif page == "Fraud Intelligence":

    page_header(
        "🕵️ Fraud Intelligence",
        "High-level indicators and patterns that may require review"
    )

    kpi_row(filtered)

    st.subheader("Risk Indicators")

    risk_rows = []

    if AMOUNT_COL and len(filtered):

        amount_threshold = filtered[AMOUNT_COL].quantile(.95)

        high_amount_fraud = filtered[
            (filtered[AMOUNT_COL] >= amount_threshold) &
            (filtered[TARGET] == 1)
        ]

        risk_rows.append({
            "Risk Indicator": "High-value fraud transactions",
            "Count": len(high_amount_fraud),
            "Threshold": f">= {amount_threshold:,.2f}"
        })

    if "Location" in filtered.columns:

        location_summary = filtered.groupby("Location").agg(
            Transactions=(TARGET, "size"),
            Fraud_Cases=(TARGET, "sum")
        )

        location_summary["Fraud Rate"] = (
            location_summary["Fraud_Cases"] /
            location_summary["Transactions"]
        )

        high_risk_locations = location_summary[
            location_summary["Fraud Rate"] >=
            location_summary["Fraud Rate"].quantile(.90)
        ]

        risk_rows.append({
            "Risk Indicator": "High fraud-rate locations",
            "Count": len(high_risk_locations),
            "Threshold": "Top 10% by observed fraud rate"
        })

    if "Merchant_Category" in filtered.columns:

        merchant_summary = filtered.groupby("Merchant_Category").agg(
            Transactions=(TARGET, "size"),
            Fraud_Cases=(TARGET, "sum")
        )

        merchant_summary["Fraud Rate"] = (
            merchant_summary["Fraud_Cases"] /
            merchant_summary["Transactions"]
        )

        high_risk_merchants = merchant_summary[
            merchant_summary["Fraud Rate"] >=
            merchant_summary["Fraud Rate"].quantile(.90)
        ]

        risk_rows.append({
            "Risk Indicator": "High fraud-rate merchant categories",
            "Count": len(high_risk_merchants),
            "Threshold": "Top 10% by observed fraud rate"
        })

    if risk_rows:

        risk_report = pd.DataFrame(risk_rows)

        st.dataframe(
            risk_report,
            use_container_width=True
        )

    else:

        st.info(
            "No supported risk indicator columns found in this dataset."
        )

    st.warning(
        "These are exploratory indicators based on historical labels. "
        "They should be reviewed before operational action."
    )


# =========================================================
# 22. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    page_header(
        "🏆 Model Comparison",
        "Compare trained model performance using available evaluation metrics"
    )

    if COMPARISON_PATH.exists():

        try:

            comparison = pd.read_csv(COMPARISON_PATH)

            st.dataframe(
                comparison,
                use_container_width=True
            )

            numeric_metrics = comparison.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_metrics:

                metric = st.selectbox(
                    "Select Metric",
                    numeric_metrics
                )

                model_col = next(
                    (
                        c for c in comparison.columns
                        if c.lower() in [
                            "model",
                            "model_name",
                            "algorithm"
                        ]
                    ),
                    None
                )

                if model_col:

                    fig = px.bar(
                        comparison,
                        x=model_col,
                        y=metric,
                        color=model_col,
                        title=f"Model Comparison: {metric}"
                    )

                    st.plotly_chart(
                        chart_style(fig),
                        use_container_width=True
                    )

            st.download_button(
                "⬇️ Download Model Comparison",
                csv_bytes(comparison),
                "model_comparison.csv",
                "text/csv"
            )

        except Exception as exc:

            st.error(
                f"Could not load model comparison: {exc}"
            )

    else:

        st.info(
            "model_comparison.csv file not found. "
            "Add the comparison file beside app.py."
        )

    if METRICS_PATH.exists():

        try:

            import json

            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                metrics = json.load(file)

            st.subheader("Model Metrics JSON")

            st.json(metrics)

        except Exception as exc:

            st.warning(f"Metrics JSON could not be read: {exc}")


# =========================================================
# 23. EXPLAINABLE AI
# =========================================================

elif page == "Explainable AI":

    page_header(
        "🧠 Explainable AI",
        "Understand model behavior and feature importance"
    )

    if MODEL_PATH.exists():

        try:

            model = joblib.load(MODEL_PATH)

            estimator = model

            if hasattr(model, "named_steps"):

                estimator = list(
                    model.named_steps.values()
                )[-1]

            if hasattr(estimator, "feature_importances_"):

                feature_names = getattr(
                    model,
                    "feature_names_in_",
                    None
                )

                importances = estimator.feature_importances_

                if feature_names is None or len(feature_names) != len(importances):

                    feature_names = [
                        f"Feature {i + 1}"
                        for i in range(len(importances))
                    ]

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(
                    "Importance",
                    ascending=False
                )

                fig = px.bar(
                    importance_df.head(20),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Feature Importances"
                )

                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(
                    chart_style(fig, 550),
                    use_container_width=True
                )

                st.dataframe(
                    importance_df,
                    use_container_width=True
                )

            else:

                st.info(
                    "The loaded model does not expose feature_importances_."
                )

        except Exception as exc:

            st.error(
                f"Explainability analysis error: {exc}"
            )

    else:

        st.warning("Model file not found.")


# =========================================================
# 24. REAL-TIME MONITORING
# =========================================================

elif page == "Real-Time Monitoring":

    page_header(
        "📡 Real-Time Monitoring",
        "Monitor the latest records available in the loaded dataset"
    )

    st.caption(
        "This page displays the latest dataset records; "
        "it does not connect to a live transaction stream."
    )

    if DATE_COL and filtered["_parsed_date"].notna().any():

        latest = filtered.sort_values(
            "_parsed_date",
            ascending=False
        ).head(100)

    else:

        latest = filtered.tail(100)

    kpi_row(latest)

    st.subheader("Latest Transactions")

    st.dataframe(
        latest.drop(
            columns=["_parsed_date"],
            errors="ignore"
        ),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Latest Transactions",
        csv_bytes(
            latest.drop(
                columns=["_parsed_date"],
                errors="ignore"
            )
        ),
        "latest_transactions.csv",
        "text/csv"
    )


# =========================================================
# 25. FRAUD FORECASTING
# =========================================================

elif page == "Fraud Forecasting":

    page_header(
        "🔮 Fraud Forecasting",
        "Explore historical fraud counts and a simple baseline projection"
    )

    if DATE_COL and filtered["_parsed_date"].notna().any():

        dated = (
            filtered.dropna(subset=["_parsed_date"])
            .set_index("_parsed_date")
        )

        daily = dated.resample("D").agg(
            Transactions=(TARGET, "size"),
            Fraud_Cases=(TARGET, "sum")
        ).reset_index()

        if len(daily) >= 7:

            daily["Day_Index"] = np.arange(len(daily))

            x = daily["Day_Index"].to_numpy()
            y = daily["Fraud_Cases"].to_numpy()

            slope, intercept = np.polyfit(x, y, 1)

            future_days = st.slider(
                "Forecast Horizon (Days)",
                min_value=7,
                max_value=60,
                value=14
            )

            future_x = np.arange(
                len(daily),
                len(daily) + future_days
            )

            forecast = np.maximum(
                0,
                slope * future_x + intercept
            )

            future_dates = pd.date_range(
                daily["_parsed_date"].max() +
                pd.Timedelta(days=1),
                periods=future_days,
                freq="D"
            )

            forecast_df = pd.DataFrame({
                "_parsed_date": future_dates,
                "Forecast Fraud Cases": forecast
            })

            historical = daily[
                ["_parsed_date", "Fraud_Cases"]
            ].rename(
                columns={"Fraud_Cases": "Historical Fraud Cases"}
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=historical["_parsed_date"],
                    y=historical["Historical Fraud Cases"],
                    mode="lines",
                    name="Historical Fraud"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_df["_parsed_date"],
                    y=forecast_df["Forecast Fraud Cases"],
                    mode="lines+markers",
                    name="Baseline Forecast"
                )
            )

            fig.update_layout(
                title="Historical Fraud and Baseline Forecast",
                xaxis_title="Date",
                yaxis_title="Fraud Cases"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

            st.dataframe(
                forecast_df,
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Forecast",
                csv_bytes(forecast_df),
                "fraud_forecast.csv",
                "text/csv"
            )

            st.warning(
                "This is a simple linear baseline, not a validated "
                "production forecasting model."
            )

        else:

            st.info(
                "At least 7 days of historical data are required."
            )

    else:

        st.info("A valid transaction date column is required.")


# =========================================================
# 26. GRAPH ANALYTICS
# =========================================================

elif page == "Graph Analytics":

    page_header(
        "🔗 Graph Analytics",
        "Explore connections between customers, locations, and merchants"
    )

    customer_col = next(
        (
            c for c in ["Customer_ID", "CustomerID"]
            if c in filtered.columns
        ),
        None
    )

    location_col = next(
        (
            c for c in ["Location", "Merchant_Category"]
            if c in filtered.columns
        ),
        None
    )

    if customer_col and location_col:

        graph_data = filtered[
            [customer_col, location_col, TARGET]
        ].dropna().copy()

        graph_data[customer_col] = (
            graph_data[customer_col].astype(str)
        )

        graph_data[location_col] = (
            graph_data[location_col].astype(str)
        )

        graph_data = graph_data.head(5000)

        connections = graph_data.groupby(
            [customer_col, location_col]
        ).agg(
            Transactions=(TARGET, "size"),
            Fraud_Cases=(TARGET, "sum")
        ).reset_index()

        connections["Fraud Rate (%)"] = (
            connections["Fraud_Cases"] /
            connections["Transactions"] * 100
        )

        st.subheader("Customer-to-Category Connections")

        st.dataframe(
            connections.sort_values(
                "Fraud_Cases",
                ascending=False
            ).head(100),
            use_container_width=True
        )

        fig = px.scatter(
            connections,
            x="Transactions",
            y="Fraud_Cases",
            size="Fraud Rate (%)",
            color="Fraud Rate (%)",
            hover_data=[customer_col, location_col],
            title="Connection-Level Fraud Activity"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Graph Connections",
            csv_bytes(connections),
            "graph_connections.csv",
            "text/csv"
        )

    else:

        st.info(
            "Customer ID and Location or Merchant Category columns "
            "are required for graph analysis."
        )


# =========================================================
# 27. DOWNLOADS & REPORTS
# =========================================================

elif page == "Downloads & Reports":

    page_header(
        "📤 Downloads & Reports",
        "Export filtered data and generate a PDF summary report"
    )

    export_data = filtered.drop(
        columns=["_parsed_date"],
        errors="ignore"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("CSV Export")

        st.download_button(
            "⬇️ Download Filtered Dataset",
            csv_bytes(export_data),
            "filtered_fraud_dataset.csv",
            "text/csv",
            use_container_width=True
        )

    with c2:

        st.subheader("PDF Summary Report")

        pdf_data = make_pdf_report(filtered)

        if pdf_data:

            st.download_button(
                "📄 Download PDF Report",
                pdf_data,
                "financial_fraud_report.pdf",
                "application/pdf",
                use_container_width=True
            )

        else:

            st.warning(
                "PDF export requires ReportLab. "
                "Install it using: pip install reportlab"
            )

    st.divider()

    st.subheader("Export Preview")

    st.dataframe(
        export_data.head(100),
        use_container_width=True
    )

    st.caption(
        f"{len(export_data):,} records are available for export."
    )


# =========================================================
# 28. AI CHAT ASSISTANT
# =========================================================

elif page == "AI Chat Assistant":

    page_header(
        "💬 AI Chat Assistant",
        "Ask questions about the loaded fraud dataset"
    )

    def chatbot_answer(question, data):

        q = question.lower().strip()

        n = len(data)

        frauds = (
            int(data[TARGET].sum())
            if n else 0
        )

        legit = n - frauds

        rate = (
            frauds / n * 100
            if n else 0
        )

        if any(word in q for word in ["total", "transactions", "records", "rows"]):

            return (
                f"The filtered dataset contains {n:,} transactions."
            )

        if "fraud rate" in q or "percentage" in q:

            return (
                f"The fraud rate in the filtered dataset is {rate:.2f}%."
            )

        if "fraud" in q or "fraudulent" in q:

            return (
                f"There are {frauds:,} transactions labelled as fraudulent."
            )

        if "legitimate" in q or "genuine" in q:

            return (
                f"There are {legit:,} transactions labelled as legitimate."
            )

        if "average" in q and AMOUNT_COL:

            average = data[AMOUNT_COL].mean()

            return (
                f"The average transaction amount is ${average:,.2f}."
            )

        if ("maximum" in q or "highest" in q) and AMOUNT_COL:

            maximum = data[AMOUNT_COL].max()

            return (
                f"The maximum transaction amount is ${maximum:,.2f}."
            )

        if "minimum" in q or "lowest" in q:

            if AMOUNT_COL:

                minimum = data[AMOUNT_COL].min()

                return (
                    f"The minimum transaction amount is ${minimum:,.2f}."
                )

        if "missing" in q or "null" in q:

            missing_count = int(
                data.isna().sum().sum()
            )

            return (
                f"There are {missing_count:,} missing values "
                f"across the filtered dataset."
            )

        if "duplicate" in q:

            duplicate_count = int(
                data.duplicated().sum()
            )

            return (
                f"The filtered dataset contains {duplicate_count:,} duplicate rows."
            )

        if "columns" in q or "features" in q:

            return (
                "Available columns: " +
                ", ".join(map(str, data.columns))
            )

        if "location" in q and "Location" in data.columns:

            return (
                "Available locations: " +
                ", ".join(
                    data["Location"]
                    .dropna()
                    .astype(str)
                    .unique()[:20]
                )
            )

        if "merchant" in q and "Merchant_Category" in data.columns:

            return (
                "Available merchant categories: " +
                ", ".join(
                    data["Merchant_Category"]
                    .dropna()
                    .astype(str)
                    .unique()[:20]
                )
            )

        if "payment" in q and "Payment_Method" in data.columns:

            return (
                "Available payment methods: " +
                ", ".join(
                    data["Payment_Method"]
                    .dropna()
                    .astype(str)
                    .unique()[:20]
                )
            )

        return (
            "I can answer questions about total transactions, fraud cases, "
            "fraud rate, legitimate transactions, average or maximum amount, "
            "missing values, duplicates, columns, locations, merchant "
            "categories, and payment methods."
        )


    if "dashboard_chat_history" not in st.session_state:

        st.session_state.dashboard_chat_history = []

    for message in st.session_state.dashboard_chat_history:

        with st.chat_message(message["role"]):

            st.write(message["content"])

    question = st.chat_input(
        "Ask something about your fraud dataset..."
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
        💳 <b>Financial Fraud Detection Dashboard</b> |
        Amdox Technologies Internship Project |
        Machine Learning & Analytics
    </div>
    """,
    unsafe_allow_html=True
)
