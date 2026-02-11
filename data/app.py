import sys
import os
import pandas as pd
import streamlit as st

# 1. Path Setup: Force Python to look at the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))

if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 2. Backend Imports with error catching
try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError as e:
    st.error(f"Module import failed. Error: {e}")

st.set_page_config(page_title="AI Risk Predictor", layout="centered")
st.title("🚀 AI-Based Software Project Risk Predictor")

# 3. File Uploaders
requirements_file = st.file_uploader("Upload Requirements (.txt)", type=["txt"])
sprint_file = st.file_uploader("Upload Sprint Tasks (.csv)", type=["csv"])

if requirements_file and sprint_file:
    if st.button("🔍 Analyze Project Risk"):
        try:
            # Ensure folders exist relative to project root
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(root_path, "results"), exist_ok=True)

            # Save uploaded files to the central data folder
            req_path = os.path.join(root_path, "data", "requirements.txt")
            with open(req_path, "w") as f:
                f.write(requirements_file.getvalue().decode("utf-8"))
            
            sprint_path = os.path.join(root_path, "data", "sprint_tasks.csv")
            pd.read_csv(sprint_file).to_csv(sprint_path, index=False)

            st.info("AI is analyzing...")
            create_combined_dataset()
            train_hybrid_model()

            # Load and Display Results
            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            if os.path.exists(res_path):
                combined = pd.read_csv(res_path)
                st.subheader("📊 Risk Analysis Results")
                st.bar_chart(combined[['ambiguity_score', 'overload_score']])
                st.success("Analysis Complete ✅")

                # Download Button
                report_path = os.path.join(root_path, "results", "ambiguity_report.csv")
                if os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button("📥 Download Report", f, "risk_report.csv", "text/csv")
            else:
                st.error("Result files not found in /results folder.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")