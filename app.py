import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from io import BytesIO
from datetime import datetime

# PDF reports use ReportLab. Install with: pip install reportlab
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# =========================================================
# PAGE CONFIGURATION & PATHS
# =========================================================
st.set_page_config(
    page_title="Financial Fraud Detection | Analytics",
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

# =========================================================
# THEME — warm cream canvas + brown sidebar, matching reference
# =========================================================
st.markdown("""
<style>
:root { --ink:#49372d; --brown:#50382c; --brown2:#765746; --cream:#f7f4eb;
        --muted:#82776d; --line:#e7dfd2; --white:#ffffff; --orange:#df8150; }
.stApp { background:var(--cream); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(247,244,235,.92); }
[data-testid="stSidebar"] { background:var(--brown); border-right:2px solid #d99a54; }
[data-testid="stSidebar"] * { color:#fff8ed !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:5px 8px; border-radius:8px; }
[data-testid="stMetric"] {
  background:#fff; border:1px solid var(--line); border-radius:13px;
  padding:16px 18px; box-shadow:0 2px 8px rgba(70,48,30,.045);
}
[data-testid="stMetricLabel"] { color:#78695e !important; font-size:13px; }
[data-testid="stMetricValue"] { color:#49372d !important; font-weight:750; }
h1,h2,h3 { color:#49372d !important; }
p, label, .stCaption { color:#76695e; }
.block-container { padding-top:1.6rem; padding-bottom:2.5rem; }
.hero {
 background:linear-gradient(115deg,#50382c,#765746 58%,#a16b4e);
 color:white; padding:25px 30px; border-radius:17px; margin-bottom:22px;
 box-shadow:0 5px 18px rgba(70,48,30,.12);
}
.hero h1 { color:white !important; margin:0; font-size:32px; }
.hero p { color:#fff1df !important; margin:7px 0 0 0; }
.section-sub { color:#8b7b6d; font-size:13px; margin-top:-8px; }
div[data-testid="stDataFrame"] { border:1px solid #e8dfd2; border-radius:10px; }
.stDownloadButton button { border-radius:9px; border:1px solid #c7ad97; }
.stButton button[kind="primary"] { background:#79543f; border-color:#79543f; }
hr { border-color:#e5dccf; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING & NORMALIZATION
# =========================================================
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.name}")
    raw = pd.read_csv(DATA_PATH)
    processed = pd.read_csv(PROCESSED_PATH) if PROCESSED_PATH.exists() else raw.copy()
    return raw, processed

try:
    df, processed = load_data()
except Exception as exc:
    st.error(f"Dataset loading error: {exc}")
    st.info("Place fraud_detection_dataset.csv in the same folder as app.py.")
    st.stop()

TARGET = "Fraudulent"
