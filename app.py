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
