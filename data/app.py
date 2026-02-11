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

st.set_page_config(page_title="AI Risk Predictor", layout="wide")
st.title("🚀 AI-Based Software Project Risk Predictor")

# 3. File Uploaders
st.sidebar.header("📁 Upload Project Files")
requirements_file = st.sidebar.file_uploader("Upload Requirements (.txt)", type=["txt"])
sprint_file = st.sidebar.file_uploader("Upload Sprint Tasks (.csv)", type=["csv"])

if requirements_file and sprint_file:
    if st.button("🔍 Run Full AI Risk Analysis", use_container_width=True):
        try:
            # Ensure folders exist
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(root_path, "results"), exist_ok=True)

            # Save uploaded files
            req_path = os.path.join(root_path, "data", "requirements.txt")
            with open(req_path, "w") as f:
                f.write(requirements_file.getvalue().decode("utf-8"))
            
            sprint_path = os.path.join(root_path, "data", "sprint_tasks.csv")
            pd.read_csv(sprint_file).to_csv(sprint_path, index=False)

            with st.spinner("AI is analyzing data and training model..."):
                create_combined_dataset()
                train_hybrid_model()

            # --- UPDATED RESULTS SECTION ---
            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            if os.path.exists(res_path):
                combined = pd.read_csv(res_path)
                
                st.markdown("---")
                st.subheader("📊 Executive Risk Summary")
                
                # Metric Layout
                cols = st.columns(len(combined))
                for i, (_, row) in enumerate(combined.iterrows()):
                    with cols[i]:
                        if row['risk_level'].lower() == "high":
                            st.error(f"Sprint {row['sprint']}: High Risk")
                        else:
                            st.success(f"Sprint {row['sprint']}: Low Risk")
                
                st.markdown("### 📈 Risk Score Distribution")
                st.bar_chart(combined[['ambiguity_score', 'overload_score']])
                
                # Download Button
                report_path = os.path.join(root_path, "results", "ambiguity_report.csv")
                if os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Detailed Risk Report (CSV)",
                            data=f,
                            file_name="risk_analysis_report.csv",
                            mime="text/csv",
                            use_container_width=True 
                        )
            else:
                st.warning("Analysis completed, but results file was not found.")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
else:
    st.info("👋 Welcome! Please upload your requirements and sprint files in the sidebar to begin.")