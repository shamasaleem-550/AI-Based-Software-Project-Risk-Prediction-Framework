import streamlit as st
import pandas as pd
import os

# Import backend logic
from src.combined_data import create_combined_dataset
from src.hybrid_risk_model import train_hybrid_model

# Page configuration
st.set_page_config(
    page_title="AI Software Project Risk Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Title
st.title("🚀 AI-Based Software Project Risk Predictor")

st.write("""
Upload your project requirement file and sprint task CSV
to analyze software project risk using AI.
""")

# Upload Section
st.subheader("📂 Upload Project Files")

requirements_file = st.file_uploader("Upload Requirements File (.txt)", type=["txt"])
sprint_file = st.file_uploader("Upload Sprint Task File (.csv)", type=["csv"])

if requirements_file and sprint_file:

    st.success("Files uploaded successfully!")

    if st.button("🔍 Analyze Project Risk"):

        # Ensure folders exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Save uploaded files
        with open("data/requirements.txt", "w") as f:
            f.write(requirements_file.getvalue().decode("utf-8"))

        sprint_df = pd.read_csv(sprint_file)
        sprint_df.to_csv("data/sprint_tasks.csv", index=False)

        st.info("Running AI analysis...")

        # Run backend logic
        create_combined_dataset()
        model = train_hybrid_model()

        # Load results
        combined = pd.read_csv("results/combined_risk_data.csv")

        st.subheader("📊 Project Risk Dashboard")

        high_count = 0

        for _, row in combined.iterrows():

            sprint = row['sprint']
            ambiguity = row['ambiguity_score']
            overload = row['overload_score']
            risk = row['risk_level']

            col1, col2, col3 = st.columns(3)

            col1.metric("Sprint", sprint)
            col2.metric("Ambiguity Score", round(ambiguity, 2))
            col3.metric("Overload Score", round(overload, 2))

            if risk == "High":
                st.error(f"🚨 Sprint {sprint} → HIGH RISK")
                high_count += 1
            elif risk == "Medium":
                st.warning(f"⚠ Sprint {sprint} → MEDIUM RISK")
            else:
                st.success(f"✅ Sprint {sprint} → LOW RISK")

            st.markdown("---")

        # Chart
        st.bar_chart(combined[['ambiguity_score', 'overload_score']])

        # Overall Status
        st.subheader("📌 Overall Project Status")

        if high_count > 0:
            st.error("Overall Project Risk: HIGH")
        else:
            st.success("Overall Project Risk: STABLE")

        st.success("Analysis Complete ✅")

