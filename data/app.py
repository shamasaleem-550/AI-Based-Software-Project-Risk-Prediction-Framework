import streamlit as st
import pandas as pd
import os

# Import your backend logic
from src.combined_data import create_combined_dataset
from src.hybrid_risk_model import train_hybrid_model

st.set_page_config(
    page_title="AI Software Project Risk Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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

        # Ensure data folders exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Save uploaded files
        with open("data/requirements.txt", "w") as f:
            f.write(requirements_file.getvalue().decode("utf-8"))

        sprint_df = pd.read_csv(sprint_file)
        sprint_df.to_csv("data/sprint_tasks.csv", index=False)

        st.info("Running AI analysis...")

        # Run backend modules
        create_combined_dataset()
        model = train_hybrid_model()

        # Load results
        combined = pd.read_csv("results/combined_risk_data.csv")

        st.subheader("📊 Risk Analysis Results")

        for _, row in combined.iterrows():

            st.metric(
                label=f"Sprint {row['sprint']} Risk Level",
                value=row['risk_level']
            )

            st.write(f"Ambiguity Score: {row['ambiguity_score']}")
            st.write(f"Overload Score: {row['overload_score']}")

        st.bar_chart(combined[['ambiguity_score', 'overload_score']])

        st.success("Analysis Complete ✅")
# Create a download button for the results
with open("results/ambiguity_report.csv", "rb") as file:
    st.download_button(
        label="📥 Download Risk Analysis Report",
        data=file,
        file_name="risk_analysis_report.csv",
        mime="text/csv"
    )