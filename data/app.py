import sys
import os
import pandas as pd
import streamlit as st

# 1. Path Setup: Add parent directory so 'src' can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 2. Backend Imports
try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError as e:
    st.error(f"Failed to import backend modules: {e}")

st.set_page_config(page_title="AI Risk Predictor", layout="centered")
st.title("🚀 AI-Based Software Project Risk Predictor")

# 3. File Uploaders
requirements_file = st.file_uploader("Upload Requirements (.txt)", type=["txt"])
sprint_file = st.file_uploader("Upload Sprint Tasks (.csv)", type=["csv"])

if requirements_file and sprint_file:
    if st.button("🔍 Analyze Project Risk"):
        try:
            # Ensure folders exist relative to project root
            os.makedirs(os.path.join(parent_dir, "data"), exist_ok=True)
            os.makedirs(os.path.join(parent_dir, "results"), exist_ok=True)

            # Save files to project root
            with open(os.path.join(parent_dir, "data/requirements.txt"), "w") as f:
                f.write(requirements_file.getvalue().decode("utf-8"))
            
            pd.read_csv(sprint_file).to_csv(os.path.join(parent_dir, "data/sprint_tasks.csv"), index=False)

            st.info("AI is analyzing...")
            create_combined_dataset()
            train_hybrid_model()

            # Load Results
            res_path = os.path.join(parent_dir, "results/combined_risk_data.csv")
            combined = pd.read_csv(res_path)

            st.subheader("📊 Risk Analysis Results")
            st.bar_chart(combined[['ambiguity_score', 'overload_score']])
            st.success("Analysis Complete ✅")

            # Download Button Logic
            report_path = os.path.join(parent_dir, "results/ambiguity_report.csv")
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    st.download_button("📥 Download Report", f, "risk_report.csv", "text/csv")

        except Exception as e:
            st.error(f"Analysis failed: {e}")