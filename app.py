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
