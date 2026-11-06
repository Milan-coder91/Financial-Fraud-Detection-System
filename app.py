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
    page_title="Financial Fraud Detection Dashboard",
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
    --brown: #50382c;
    --brown2: #79543f;
    --cream: #f7f4eb;
    --orange: #df8150;
    --red: #c94f45;
    --green: #39856b;
    --line: #e7dfd2;
}

.stApp {
    background-color: var(--cream);
    color: #49372d;
}

[data-testid="stHeader"] {
    background: rgba(247,244,235,0.95);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #50382c;
    border-right: 2px solid #d99a54;
}

[data-testid="stSidebar"] * {
    color: #fff8ed !important;
}

/* Sidebar navigation */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 7px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent !important;
    border-radius: 8px;
    padding: 7px 9px;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.15) !important;
}

/* Remove radio indicator */
[data-testid="stSidebar"] [role="radio"] > div:first-child {
    display: none !important;
}

/* Selected navigation item */
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
    background: #a16b4e !important;
    border-radius: 8px;
    font-weight: 700;
}

/* Checkbox */
[data-testid="stCheckbox"] input:checked + div {
    background-color: #79543f !important;
    border-color: #79543f !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    border-radius: 9px;
    border: 1px solid #c7ad97;
    transition: 0.2s;
}

.stButton button[kind="primary"] {
    background: #79543f !important;
    border-color: #79543f !important;
    color: white !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    border-color: #79543f !important;
}

/* KPI Cards */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e7dfd2;
    border-radius: 13px;
    padding: 16px 18px;
    box-shadow: 0 3px 12px rgba(70,48,30,0.06);
}

[data-testid="stMetricLabel"] {
    color: #78695e !important;
}

[data-testid="stMetricValue"] {
    color: #49372d !important;
    font-weight: 750;
}

/* Headings */
h1, h2, h3 {
    color: #49372d !important;
}

p, label {
    color: #76695e;
}

.section-sub {
    color: #8b7b6d;
    font-size: 13px;
    margin-top: -8px;
}

/* Hero */
.hero {
    background: linear-gradient(115deg,#50382c,#765746 58%,#a16b4e);
    color: white;
    padding: 28px 30px;
    border-radius: 17px;
    margin-bottom: 22px;
}

.hero h1 {
    color: white !important;
    margin: 0;
    font-size: 32px;
}

.hero p {
    color: #fff1df !important;
    margin: 7px 0 0 0;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border: 1px solid #e8dfd2;
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #e5dccf;
}

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

BROWN = "#79543f"
ORANGE = "#df8150"
RED = "#c94f45"
GREEN = "#39856b"
BLUE = "#477da8"

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
        template="plotly_white",
        paper_bgcolor="#f7f4eb",
        plot_bgcolor="#ffffff",
        font=dict(
            color="#49372d",
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
        linecolor="#e8dfd2"
    )

    fig.update_yaxes(
        gridcolor="#eee8df",
        zerolinecolor="#eee8df"
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
            "Financial Fraud Detection System",
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

st.sidebar.markdown("# 💳 Financial Fraud Dashboard")

st.sidebar.caption("AMDOX TECHNOLOGIES | INTERNSHIP PROJECT")

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
    label_visibility="collapsed"
)

page = pages[selected_page]

st.sidebar.divider()

st.sidebar.markdown("### 🎛️ Dashboard Filters")

filtered = filtered_frame()

st.sidebar.caption(
    f"Showing {len(filtered):,} of {total:,} transactions"
)

st.sidebar.divider()

st.sidebar.caption("Financial Fraud Detection | Analytics Platform")


# =========================================================
# 8. EXECUTIVE DASHBOARD
# =========================================================

if page == "Executive Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>💳 Financial Fraud Detection Dashboard</h1>
        <p>Enterprise Transaction Monitoring • Fraud Analytics • Risk Intelligence</p>
        <p>Amdox Technologies Internship Project</p>
    </div>
    """, unsafe_allow_html=True)

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

    left, right = st.columns(2)

    with left:

        st.subheader("Fraud Distribution")

        f = int(filtered[TARGET].sum()) if len(filtered) else 0
        l = len(filtered) - f

        fig = px.pie(
            names=["Legitimate", "Fraudulent"],
            values=[l, f],
            hole=.58,
            color=["Legitimate", "Fraudulent"],
            color_discrete_map={
                "Legitimate": GREEN,
                "Fraudulent": RED
            }
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with right:

        st.subheader("Transaction Amount Distribution")

        if AMOUNT_COL:

            fig = px.histogram(
                filtered,
                x=AMOUNT_COL,
                color=TARGET,
                nbins=35,
                barmode="overlay",
                opacity=.78,
                color_discrete_map={
                    0: BLUE,
                    1: RED
                }
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        else:
            st.info("Transaction amount field not found.")

    st.subheader("📊 Fraud Analytics")

    available_categories = [
        c for c in [
            "Location",
            "Merchant_Category",
            "Payment_Method",
            "Device_Type"
        ]
        if c in filtered.columns
    ]

    if available_categories:

        for i in range(0, len(available_categories), 2):

            c1, c2 = st.columns(2)

            with c1:
                category_chart(
                    filtered,
                    available_categories[i],
                    f"Fraud by {available_categories[i]}"
                )

            if i + 1 < len(available_categories):

                with c2:
                    category_chart(
                        filtered,
                        available_categories[i + 1],
                        f"Fraud by {available_categories[i + 1]}"
                    )

    if DATE_COL and filtered["_parsed_date"].notna().any():

        st.subheader("📈 Transaction Trend")

        dated = (
            filtered.dropna(subset=["_parsed_date"])
            .set_index("_parsed_date")
        )

        daily = dated.resample("D").agg(
            Transactions=(TARGET, "size"),
            Fraud_Cases=(TARGET, "sum")
        ).reset_index()

        fig = px.line(
            daily,
            x="_parsed_date",
            y=["Transactions", "Fraud_Cases"],
            markers=True,
            title="Daily Transaction and Fraud Trends"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )


# =========================================================
# 9. OVERVIEW
# =========================================================

elif page == "Overview":

    page_header(
        "📊 Executive Overview",
        "Key indicators and high-level transaction health"
    )

    kpi_row(filtered)

    st.subheader("Dataset Summary")

    f = int(filtered[TARGET].sum()) if len(filtered) else 0

    st.write(
        f"Dataset contains **{len(filtered):,} transactions**, "
        f"including **{f:,} fraud records**."
    )

    st.write(
        f"Fraud share: **{f / len(filtered) * 100:.2f}%**."
        if len(filtered) else "No records found."
    )

    st.dataframe(
        filtered.drop(columns=["_parsed_date"], errors="ignore").head(100),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Dataset",
        csv_bytes(filtered.drop(columns=["_parsed_date"], errors="ignore")),
        "financial_fraud_dataset.csv",
        "text/csv"
    )


# =========================================================
# 10. TREND ANALYSIS
# =========================================================

elif page == "Trend Analysis":

    page_header(
        "📈 Trend Analysis",
        "Explore transaction volume and fraud activity over time"
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

        daily["Fraud Rate (%)"] = np.where(
            daily["Transactions"] > 0,
            daily["Fraud_Cases"] / daily["Transactions"] * 100,
            0
        )

        c1, c2 = st.columns(2)

        with c1:

            fig = px.line(
                daily,
                x="_parsed_date",
                y="Transactions",
                markers=True,
                title="Daily Transactions"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        with c2:

            fig = px.line(
                daily,
                x="_parsed_date",
                y="Fraud_Cases",
                markers=True,
                title="Daily Fraud Cases"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        fig = px.area(
            daily,
            x="_parsed_date",
            y="Fraud Rate (%)",
            title="Daily Fraud Rate"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

        st.dataframe(daily, use_container_width=True)

        st.download_button(
            "⬇️ Download Daily Trend",
            csv_bytes(daily),
            "daily_fraud_trend.csv",
            "text/csv"
        )

    else:
        st.warning("Valid transaction date column required.")


# =========================================================
# 11. LOCATION FRAUD
# =========================================================

elif page == "Location Fraud":

    page_header(
        "📍 Location Fraud Analysis",
        "Compare transaction activity and fraud risk across locations"
    )

    if "Location" in filtered.columns:

        category_chart(
            filtered,
            "Location",
            "Fraud Cases by Location",
            30
        )

        loc = filtered.groupby("Location").agg(
            Transactions=(TARGET, "size"),
            Fraud_Cases=(TARGET, "sum")
        ).reset_index()

        loc["Fraud Rate (%)"] = (
            loc["Fraud_Cases"] / loc["Transactions"] * 100
        )

        fig = px.scatter(
            loc,
            x="Transactions",
            y="Fraud Rate (%)",
            size="Fraud_Cases",
            color="Fraud Rate (%)",
            hover_name="Location",
            title="Location Risk Analysis",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Location Report",
            csv_bytes(loc),
            "location_fraud_analysis.csv",
            "text/csv"
        )

    else:
        st.info("Location column not available.")


# =========================================================
# 12. TRANSACTION AMOUNT
# =========================================================

elif page == "Transaction Amount":

    page_header(
        "💰 Transaction Amount Analysis",
        "Understand spending distribution and fraud-related amounts"
    )

    if AMOUNT_COL:

        cols = st.columns(4)

        cols[0].metric(
            "Total Amount",
            f"${filtered[AMOUNT_COL].sum():,.2f}"
        )

        cols[1].metric(
            "Average Amount",
            f"${filtered[AMOUNT_COL].mean():,.2f}"
        )

        cols[2].metric(
            "Highest Amount",
            f"${filtered[AMOUNT_COL].max():,.2f}"
        )

        fraud_amount_filtered = filtered.loc[
            filtered[TARGET] == 1,
            AMOUNT_COL
        ].sum()

        cols[3].metric(
            "Fraud Amount",
            f"${fraud_amount_filtered:,.2f}"
        )

        c1, c2 = st.columns(2)

        with c1:

            fig = px.histogram(
                filtered,
                x=AMOUNT_COL,
                color=TARGET,
                nbins=45,
                barmode="overlay",
                title="Amount Distribution",
                color_discrete_map={0: BLUE, 1: RED}
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        with c2:

            fig = px.box(
                filtered,
                x=TARGET,
                y=AMOUNT_COL,
                color=TARGET,
                title="Amount by Transaction Class",
                color_discrete_map={0: BLUE, 1: RED}
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        amount_summary = (
            filtered.groupby(TARGET)[AMOUNT_COL]
            .agg(["count", "mean", "median", "min", "max", "sum"])
            .reset_index()
        )

        st.dataframe(
            amount_summary,
            use_container_width=True
        )

    else:
        st.info("Transaction amount column not found.")


# =========================================================
# 13. MERCHANT ANALYSIS
# =========================================================

elif page == "Merchant Analysis":

    page_header(
        "🏪 Merchant Analysis",
        "Identify merchant categories with higher fraud counts and rates"
    )

    if "Merchant_Category" in filtered.columns:

        category_chart(
            filtered,
            "Merchant_Category",
            "Fraud by Merchant Category"
        )

        merchant = filtered.groupby("Merchant_Category").agg(
            Transactions=(TARGET, "size"),
            Fraud_Cases=(TARGET, "sum")
        ).reset_index()

        merchant["Fraud Rate (%)"] = (
            merchant["Fraud_Cases"] / merchant["Transactions"] * 100
        )

        st.dataframe(
            merchant.sort_values(
                "Fraud Rate (%)",
                ascending=False
            ),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Merchant Report",
            csv_bytes(merchant),
            "merchant_fraud_report.csv",
            "text/csv"
        )

    else:
        st.info("Merchant_Category column not found.")


# =========================================================
# 14. TRANSACTION TYPE
# =========================================================

elif page == "Transaction Type":

    page_header(
        "💳 Transaction Type Analysis",
        "Analyze fraud patterns across transaction and payment types"
    )

    candidates = [
        c for c in [
            "Payment_Method",
            "Transaction_Type",
            "TransactionType",
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

        fig = px.bar(
            monthly,
            x="Month",
            y="Fraud Cases",
            title="Fraud Cases by Month"
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    else:
        st.warning("Valid transaction date required.")


# =========================================================
# 16. DETAILED INVESTIGATION
# =========================================================

elif page == "Detailed Investigation":

    page_header(
        "🔎 Detailed Investigation",
        "Search and filter suspicious transactions"
    )

    customer_col = next(
        (
            c for c in ["Customer_ID", "CustomerID"]
            if c in df.columns
        ),
        None
    )

    transaction_col = next(
        (
            c for c in ["Transaction_ID", "TransactionID"]
            if c in df.columns
        ),
        None
    )

    c1, c2 = st.columns(2)

    with c1:
        customer_query = (
            st.text_input("Search Customer ID")
            if customer_col else ""
        )

    with c2:
        transaction_query = (
            st.text_input("Search Transaction ID")
            if transaction_col else ""
        )

    only_fraud = st.checkbox("🚨 Only Fraud Transactions")

    high_amount = (
        st.checkbox("💰 Only High Amount Transactions")
        if AMOUNT_COL else False
    )

    investigation = filtered.copy()

    if customer_col and customer_query:

        investigation = investigation[
            investigation[customer_col].astype(str).str.contains(
                customer_query,
                case=False,
                na=False
            )
        ]

    if transaction_col and transaction_query:

        investigation = investigation[
            investigation[transaction_col].astype(str).str.contains(
                transaction_query,
                case=False,
                na=False
            )
        ]

    if only_fraud:
        investigation = investigation[
            investigation[TARGET] == 1
        ]

    if high_amount and AMOUNT_COL:

        cutoff = df[AMOUNT_COL].quantile(.90)

        investigation = investigation[
            investigation[AMOUNT_COL] >= cutoff
        ]

    m = st.columns(4)

    m[0].metric(
        "Matching Transactions",
        f"{len(investigation):,}"
    )

    m[1].metric(
        "Fraud Cases",
        f"{int(investigation[TARGET].sum()):,}"
    )

    m[2].metric(
        "Total Amount",
        f"${investigation[AMOUNT_COL].sum():,.2f}"
        if AMOUNT_COL else "N/A"
    )

    m[3].metric(
        "Average Amount",
        f"${investigation[AMOUNT_COL].mean():,.2f}"
        if AMOUNT_COL and len(investigation) else "N/A"
    )

    st.dataframe(
        investigation.drop(
            columns=["_parsed_date"],
            errors="ignore"
        ),
        use_container_width=True,
        height=420
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
        "🤖 Fraud Prediction Center",
        "Screen a new transaction using the saved machine learning model"
    )

    if not MODEL_PATH.exists():

        st.error(
            f"Model file not found: {MODEL_PATH.name}"
        )

    else:

        try:

            model = joblib.load(MODEL_PATH)

            st.info(
                "Prediction is a model-based screening estimate, "
                "not a confirmed fraud verdict."
            )

            with st.form("fraud_prediction_form"):

                c1, c2 = st.columns(2)

                with c1:

                    amount = st.number_input(
                        "Transaction Amount",
                        min_value=0.0,
                        value=100.0
                    )

                    merchant = (
                        st.selectbox(
                            "Merchant Category",
                            sorted(
                                df["Merchant_Category"]
                                .dropna().astype(str).unique()
                            )
                        )
                        if "Merchant_Category" in df.columns else ""
                    )

                    payment = (
                        st.selectbox(
                            "Payment Method",
                            sorted(
                                df["Payment_Method"]
                                .dropna().astype(str).unique()
                            )
                        )
                        if "Payment_Method" in df.columns else ""
                    )

                    device = (
                        st.selectbox(
                            "Device Type",
                            sorted(
                                df["Device_Type"]
                                .dropna().astype(str).unique()
                            )
                        )
                        if "Device_Type" in df.columns else ""
                    )

                    location = (
                        st.selectbox(
                            "Location",
                            sorted(
                                df["Location"]
                                .dropna().astype(str).unique()
                            )
                        )
                        if "Location" in df.columns else ""
                    )

                    international = st.selectbox(
                        "International Transaction",
                        [0, 1]
                    )

                with c2:

                    previous = st.number_input(
                        "Previous Transactions",
                        min_value=0,
                        value=5
                    )

                    average = st.number_input(
                        "Average Spend",
                        min_value=0.0,
                        value=100.0
                    )

                    age = st.number_input(
                        "Account Age (Days)",
                        min_value=0,
                        value=365
                    )

                    keyword = (
                        st.selectbox(
                            "Suspicious Keyword",
                            sorted(
                                df["Suspicious_Keyword"]
                                .dropna().astype(str).unique()
                            )
                        )
                        if "Suspicious_Keyword" in df.columns else ""
                    )

                    threshold = st.slider(
                        "Fraud Alert Threshold",
                        .05,
                        .95,
                        .50,
                        .05
                    )

                submit = st.form_submit_button(
                    "🔍 Analyze Transaction",
                    type="primary"
                )

            if submit:

                now = pd.Timestamp.now()

                row = {
                    "Transaction_ID": "NEW_TRANSACTION",
                    "Customer_ID": "NEW_CUSTOMER",
                    "Transaction_Date": now.strftime("%d-%m-%Y %H:%M"),
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
                    "Transaction_Hour": now.hour,
                    "Transaction_Day": now.day,
                    "Transaction_Month": now.month,
                    "Transaction_Weekday": now.weekday(),
                    "Amount_to_Average_Ratio": (
                        amount / average if average > 0 else 0
                    ),
                    "Above_Average_Spend": int(amount > average),
                    "International_Status": (
                        "International" if international else "Domestic"
                    )
                }

                prediction_input = pd.DataFrame([row])

                expected = getattr(
                    model,
                    "feature_names_in_",
                    None
                )

                if expected is not None:

                    prediction_input = prediction_input.reindex(
                        columns=expected
                    )

                try:

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(
                            prediction_input
                        )[0]

                        classes = list(
                            getattr(model, "classes_", [0, 1])
                        )

                        fraud_index = (
                            classes.index(1)
                            if 1 in classes else -1
                        )

                        probability = (
                            float(probabilities[fraud_index])
                            if fraud_index >= 0
                            else float(probabilities[-1])
                        )

                    else:

                        prediction = int(
                            model.predict(prediction_input)[0]
                        )

                        probability = float(prediction)

                    st.metric(
                        "Fraud Probability",
                        f"{probability * 100:.2f}%"
                    )

                    if probability >= threshold:

                        st.error(
                            "⚠️ Flagged for Fraud Review"
                        )

                    else:

                        st.success(
                            "✅ Below Selected Fraud Alert Threshold"
                        )

                except Exception as exc:

                    st.error(
                        f"Prediction could not be completed: {exc}"
                    )

                    st.caption(
                        "Input columns aur model training pipeline "
                        "match hone chahiye."
                    )

        except Exception as exc:

            st.error(f"Model loading error: {exc}")


# =========================================================
# 18. DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":

    page_header(
        "📁 Dataset Overview",
        "Dataset structure, quality checks, and statistical summary"
    )

    a, b, c, d = st.columns(4)

    a.metric("Rows", f"{filtered.shape[0]:,}")
    b.metric("Columns", filtered.shape[1])
    c.metric("Fraud Cases", f"{int(filtered[TARGET].sum()):,}")
    d.metric("Duplicate Rows", f"{int(filtered.duplicated().sum()):,}")

    clean_df = filtered.drop(
        columns=["_parsed_date"],
        errors="ignore"
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        clean_df.head(100),
        use_container_width=True
    )

    quality = pd.DataFrame({
        "Column": clean_df.columns,
        "Data Type": clean_df.dtypes.astype(str).values,
        "Missing Values": clean_df.isnull().sum().values,
        "Unique Values": clean_df.nunique().values
    })

    st.subheader("Data Quality")

    st.dataframe(
        quality,
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        clean_df.describe(include="all").T,
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Data Quality Report",
        csv_bytes(quality),
        "data_quality_report.csv",
        "text/csv"
    )


# =========================================================
# 19. EDA DASHBOARD
# =========================================================

elif page == "EDA Dashboard":

    page_header(
        "📉 Exploratory Data Analysis",
        "Distribution, category patterns, and numerical relationships"
    )

    c1, c2 = st.columns(2)

    with c1:

        f = int(filtered[TARGET].sum()) if len(filtered) else 0

        fig = px.pie(
            names=["Legitimate", "Fraudulent"],
            values=[len(filtered) - f, f],
            hole=.5,
            title="Fraud Class Distribution",
            color_discrete_sequence=[GREEN, RED]
        )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

    with c2:

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

    engineered = [
        c for c in [
            "Transaction_Hour",
            "Transaction_Day",
            "Transaction_Month",
            "Transaction_Weekday",
            "Amount_to_Average_Ratio",
            "Above_Average_Spend",
            "International_Status"
        ]
        if c in processed.columns
    ]

    if engineered:

        selected = st.selectbox(
            "Select Feature",
            engineered
        )

        if pd.api.types.is_numeric_dtype(processed[selected]):

            fig = px.histogram(
                processed,
                x=selected,
                nbins=30,
                title=f"{selected} Distribution"
            )

        else:

            counts = (
                processed[selected]
                .value_counts(dropna=False)
                .reset_index()
            )

            counts.columns = [selected, "Count"]

            fig = px.bar(
                counts,
                x=selected,
                y="Count",
                title=f"{selected} Distribution"
            )

        st.plotly_chart(
            chart_style(fig),
            use_container_width=True
        )

        st.dataframe(
            processed[engineered].head(100),
            use_container_width=True
        )

    else:

        st.warning(
            "Expected engineered feature columns not found."
        )

    st.download_button(
        "⬇️ Download Processed Dataset",
        csv_bytes(processed),
        "processed_fraud_dataset.csv",
        "text/csv"
    )


# =========================================================
# 21. FRAUD INTELLIGENCE
# =========================================================

elif page == "Fraud Intelligence":

    page_header(
        "🕵️ Fraud Intelligence Center",
        "Investigate fraud cases and suspicious transaction patterns"
    )

    fraud_filtered = filtered[
        filtered[TARGET] == 1
    ]

    c = st.columns(4)

    c[0].metric(
        "Fraud Cases",
        f"{len(fraud_filtered):,}"
    )

    c[1].metric(
        "Fraud Rate",
        f"{len(fraud_filtered) / len(filtered) * 100:.2f}%"
        if len(filtered) else "0%"
    )

    c[2].metric(
        "Fraud Amount",
        f"${fraud_filtered[AMOUNT_COL].sum():,.2f}"
        if AMOUNT_COL else "N/A"
    )

    c[3].metric(
        "Average Fraud Amount",
        f"${fraud_filtered[AMOUNT_COL].mean():,.2f}"
        if AMOUNT_COL and len(fraud_filtered) else "N/A"
    )

    for col in [
        "Is_International",
        "Merchant_Category",
        "Payment_Method",
        "Device_Type",
        "Location"
    ]:

        if col in filtered.columns:

            category_chart(
                filtered,
                col,
                f"Fraud Intelligence: {col}"
            )

    st.subheader("Fraud Transaction Records")

    st.dataframe(
        fraud_filtered.drop(
            columns=["_parsed_date"],
            errors="ignore"
        ),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Fraud Cases",
        csv_bytes(
            fraud_filtered.drop(
                columns=["_parsed_date"],
                errors="ignore"
            )
        ),
        "fraud_transactions.csv",
        "text/csv"
    )


# =========================================================
# 22. MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    page_header(
        "🏆 Model Comparison",
        "Compare classification models evaluated during training"
    )

    if COMPARISON_PATH.exists():

        results = pd.read_csv(COMPARISON_PATH)

        st.dataframe(
            results,
            use_container_width=True
        )

        metric_candidates = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC"
        ]

        available = [
            c for c in metric_candidates
            if c in results.columns
        ]

        if available:

            metric = st.selectbox(
                "Select Evaluation Metric",
                available
            )

            fig = px.bar(
                results,
                x=results.columns[0],
                y=metric,
                color=metric,
                title=f"Model Comparison — {metric}",
                color_continuous_scale="Oranges"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        st.download_button(
            "⬇️ Download Model Comparison",
            csv_bytes(results),
            "model_comparison.csv",
            "text/csv"
        )

    else:

        st.warning("model_comparison.csv was not found.")


# =========================================================
# 23. EXPLAINABLE AI
# =========================================================

elif page == "Explainable AI":

    page_header(
        "🧠 Explainable AI",
        "Inspect model feature coefficients or feature importance"
    )

    if MODEL_PATH.exists():

        try:

            model = joblib.load(MODEL_PATH)

            estimator = (
                model.steps[-1][1]
                if hasattr(model, "steps")
                else model
            )

            importance = None

            if hasattr(estimator, "feature_importances_"):

                importance = estimator.feature_importances_

            elif hasattr(estimator, "coef_"):

                importance = np.abs(
                    estimator.coef_
                ).ravel()

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

                    names = [
                        f"Feature {i+1}"
                        for i in range(len(importance))
                    ]

                explain = pd.DataFrame({
                    "Feature": names,
                    "Importance": importance
                }).sort_values(
                    "Importance",
                    ascending=False
                ).head(20)

                fig = px.bar(
                    explain.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top Model Features",
                    color="Importance",
                    color_continuous_scale="Oranges"
                )

                st.plotly_chart(
                    chart_style(fig, 520),
                    use_container_width=True
                )

                st.dataframe(
                    explain,
                    use_container_width=True
                )

                st.caption(
                    "Logistic Regression ke absolute coefficient "
                    "magnitudes causal explanations nahi hain."
                )

            else:

                st.info(
                    "Estimator native feature importance "
                    "ya coefficients expose nahi karta."
                )

        except Exception as exc:

            st.warning(
                f"Model explanation unavailable: {exc}"
            )

    else:

        st.warning("Saved model file not found.")


# =========================================================
# 24. REAL-TIME MONITORING
# =========================================================

elif page == "Real-Time Monitoring":

    page_header(
        "📡 Transaction Monitoring",
        "Review the latest records present in the historical dataset"
    )

    st.info(
        "Ye page dataset records monitor karta hai. "
        "Ye live bank transaction stream se connected nahi hai."
    )

    monitor = filtered.copy()

    if DATE_COL:

        monitor = monitor.sort_values(
            "_parsed_date",
            ascending=False,
            na_position="last"
        )

    c = st.columns(3)

    c[0].metric(
        "Transactions",
        f"{len(monitor):,}"
    )

    c[1].metric(
        "Fraud Records",
        f"{int(monitor[TARGET].sum()):,}"
    )

    c[2].metric(
        "Fraud Rate",
        f"{monitor[TARGET].mean() * 100:.2f}%"
        if len(monitor) else "0%"
    )

    st.subheader("Recent Transactions")

    st.dataframe(
        monitor.drop(
            columns=["_parsed_date"],
            errors="ignore"
        ).head(100),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Monitoring Records",
        csv_bytes(
            monitor.drop(
                columns=["_parsed_date"],
                errors="ignore"
            )
        ),
        "monitoring_records.csv",
        "text/csv"
    )


# =========================================================
# 25. FRAUD FORECASTING
# =========================================================

elif page == "Fraud Forecasting":

    page_header(
        "🔮 Fraud Trend & Forecasting",
        "Historical fraud activity and transaction volume"
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

        c1, c2 = st.columns(2)

        with c1:

            fig = px.line(
                daily,
                x="_parsed_date",
                y="Fraud_Cases",
                markers=True,
                title="Historical Fraud Cases"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        with c2:

            fig = px.area(
                daily,
                x="_parsed_date",
                y="Transactions",
                title="Daily Transaction Volume"
            )

            st.plotly_chart(
                chart_style(fig),
                use_container_width=True
            )

        st.info(
            "Ye historical trends hain. Future fraud predictions ke liye "
            "separately trained and validated forecasting model required hai."
        )

        st.dataframe(
            daily,
            use_container_width=True
        )

    else:

        st.warning("Valid date column required.")


# =========================================================
# 26. GRAPH ANALYTICS
# =========================================================

elif page == "Graph Analytics":

    page_header(
        "🔗 Graph & Correlation Analytics",
        "Explore numerical relationships in the transaction dataset"
    )

    numeric = filtered.select_dtypes(
        include=np.number
    )

    if numeric.shape[1] >= 2:

        corr_data = (
            numeric.sample(
                min(len(numeric), 50000),
                random_state=42
            )
            if len(numeric) > 50000
            else numeric
        )

        corr = corr_data.corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap"
        )

        st.plotly_chart(
            chart_style(fig, 600),
            use_container_width=True
        )

        if TARGET in corr.columns:

            target_corr = (
                corr[TARGET]
                .drop(TARGET)
                .sort_values()
                .reset_index()
            )

            target_corr.columns = [
                "Feature",
                "Correlation with Fraud"
            ]

            fig = px.bar(
                target_corr,
                x="Correlation with Fraud",
                y="Feature",
                orientation="h",
                title="Feature Correlation with Fraud",
                color="Correlation with Fraud",
                color_continuous_scale="RdBu_r"
            )

            st.plotly_chart(
                chart_style(fig, 500),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Correlation Results",
                csv_bytes(target_corr),
                "fraud_correlations.csv",
                "text/csv"
            )

    else:

        st.info(
            "Not enough numerical columns for correlation analysis."
        )


# =========================================================
# 27. DOWNLOADS & REPORTS
# =========================================================

elif page == "Downloads & Reports":

    page_header(
        "📤 Downloads & Reports",
        "Export datasets, filtered results, fraud records, and PDF summary"
    )

    clean_df = df.drop(
        columns=["_parsed_date"],
        errors="ignore"
    )

    clean_processed = processed.drop(
        columns=["_parsed_date"],
        errors="ignore"
    )

    clean_fraud = fraud_df.drop(
        columns=["_parsed_date"],
        errors="ignore"
    )

    d1, d2, d3 = st.columns(3)

    d1.download_button(
        "⬇️ Full Dataset CSV",
        csv_bytes(clean_df),
        "full_fraud_dataset.csv",
        "text/csv",
        use_container_width=True
    )

    d2.download_button(
        "⬇️ Processed Dataset CSV",
        csv_bytes(clean_processed),
        "processed_fraud_dataset.csv",
        "text/csv",
        use_container_width=True
    )

    d3.download_button(
        "⬇️ Fraud Records CSV",
        csv_bytes(clean_fraud),
        "fraud_records.csv",
        "text/csv",
        use_container_width=True
    )

    st.divider()

    st.subheader("Filtered Dataset Export")

    st.write(
        f"Current sidebar filters return **{len(filtered):,} records**."
    )

    st.download_button(
        "⬇️ Download Filtered Records",
        csv_bytes(
            filtered.drop(
                columns=["_parsed_date"],
                errors="ignore"
            )
        ),
        "filtered_transactions.csv",
        "text/csv"
    )

    st.divider()

    st.subheader("📄 PDF Summary Report")

    if REPORTLAB_AVAILABLE:

        pdf = make_pdf_report(filtered)

        st.download_button(
            "📄 Download PDF Report",
            pdf,
            "financial_fraud_summary_report.pdf",
            "application/pdf",
            type="primary"
        )

    else:

        st.warning(
            "ReportLab installed nahi hai. "
            "requirements.txt mein reportlab add karo."
        )

    st.divider()

    st.subheader("Model Artifacts")

    if COMPARISON_PATH.exists():

        st.download_button(
            "⬇️ Model Comparison CSV",
            COMPARISON_PATH.read_bytes(),
            "model_comparison.csv",
            "text/csv"
        )

    if METRICS_PATH.exists():

        st.download_button(
            "⬇️ Model Metrics JSON",
            METRICS_PATH.read_bytes(),
            "model_metrics.json",
            "application/json"
        )

    if MODEL_PATH.exists():

        st.success(
            f"Saved model found: {MODEL_PATH.name}"
        )

        st.caption(
            "Model server par prediction ke liye available hai."
        )


# =========================================================
# 28. OFFLINE AI CHAT ASSISTANT
# =========================================================

elif page == "AI Chat Assistant":

    page_header(
        "💬 Dashboard Chat Assistant",
        "Ask questions about your selected fraud dataset"
    )

    st.info(
        "Offline rule-based assistant. API key ki zaroorat nahi hai."
    )

    st.subheader("📊 Dataset Quick Summary")

    kpi_row(filtered)

    if "dashboard_chat_history" not in st.session_state:

        st.session_state.dashboard_chat_history = []

    def chatbot_answer(question, data):

        q = question.lower().strip()

        n = len(data)

        frauds = int(data[TARGET].sum()) if n else 0

        legitimate = n - frauds

        rate = frauds / n * 100 if n else 0

        if any(x in q for x in ["fraud rate", "percentage", "percent"]):

            return f"Fraud rate is {rate:.2f}%."

        if any(x in q for x in ["total transaction", "total records", "total data"]):

            return f"Total transactions: {n:,}."

        if any(x in q for x in ["how many fraud", "fraud cases", "total fraud"]):

            return f"Total fraud transactions: {frauds:,}."

        if any(x in q for x in ["normal", "legitimate"]):

            return f"Legitimate transactions: {legitimate:,}."

        if any(x in q for x in ["average amount", "mean amount"]):

            if AMOUNT_COL:
                return (
                    f"Average transaction amount: "
                    f"${data[AMOUNT_COL].mean():,.2f}."
                )

            return "Amount column not available."

        if any(x in q for x in ["maximum amount", "highest amount", "max amount"]):

            if AMOUNT_COL:
                return (
                    f"Maximum transaction amount: "
                    f"${data[AMOUNT_COL].max():,.2f}."
                )

            return "Amount column not available."

        if "missing" in q or "null" in q:

            missing = int(data.isnull().sum().sum())

            return f"Total missing values: {missing:,}."

        if "duplicate" in q:

            duplicates = int(
                data.drop(
                    columns=["_parsed_date"],
                    errors="ignore"
                ).duplicated().sum()
            )

            return f"Duplicate rows: {duplicates:,}."

        if "column" in q:

            return "Dataset columns: " + ", ".join(
                data.drop(
                    columns=["_parsed_date"],
                    errors="ignore"
                ).columns.astype(str)
            )

        if "location" in q and "Location" in data.columns:

            return (
                "Available locations: "
                + ", ".join(
                    data["Location"]
                    .dropna()
                    .astype(str)
                    .unique()[:15]
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
        💳 <b>Financial Fraud Detection Dashboard</b> |
        Amdox Technologies Internship Project |
        Machine Learning & Analytics
    </div>
    """,
    unsafe_allow_html=True
)
