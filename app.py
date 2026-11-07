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
