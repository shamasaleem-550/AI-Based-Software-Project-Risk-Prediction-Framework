import sys
import os
import pandas as pd
import streamlit as st

# --- 1. SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))

if root_path not in sys.path:
    sys.path.insert(0, root_path)

# --- 2. BACKEND IMPORTS ---
try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError as e:
    st.error(f"Module import failed. Please check folder structure. Error: {e}")

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Project Risk Predictor", layout="wide")

# Professional UI Styling
st.markdown("""
    <style>
    .stMetric { 
        background-color: #1E1E1E !important; 
        color: white !important;
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #3E3E3E;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #AAAAAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI-Based Software Project Risk Predictor")
st.markdown(f"**Developed by SHAMA SALEEM** | Framework for NLP & Resource Risk Analysis")

# --- 4. SIDEBAR - FILE UPLOADS & TEMPLATES ---
with st.sidebar:
    st.header("📁 Data Input")
    requirements_file = st.file_uploader("1. Upload Requirements (.txt)", type=["txt"])
    sprint_file = st.file_uploader("2. Upload Sprint Tasks (.csv)", type=["csv"])
    
    st.markdown("---")
    st.subheader("💡 Don't have files?")
    st.caption("Use these templates to test the AI logic:")
    
    sample_req = "The system must be fast and robust. We need flexible modules."
    st.download_button("📥 Sample Requirements (.txt)", sample_req, "sample_req.txt")

    sample_csv = "sprint,task_name,hours_assigned,developer_capacity\n1,Backend,45,40\n2,Frontend,20,40\n3,API,55,40"
    st.download_button("📥 Sample Sprint Data (.csv)", sample_csv, "sample_sprint.csv")
    
    st.markdown("---")
    st.info("Analysis detects ambiguity in text and workload overload.")

# --- 5. MAIN LOGIC ---
if requirements_file and sprint_file:
    if st.button("🔍 Run Full AI Risk Analysis", use_container_width=True):
        try:
            # Ensure directories exist
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(root_path, "results"), exist_ok=True)

            # Save uploaded files
            with open(os.path.join(root_path, "data", "requirements.txt"), "wb") as f:
                f.write(requirements_file.getvalue())
            with open(os.path.join(root_path, "data", "sprint_tasks.csv"), "wb") as f:
                f.write(sprint_file.getvalue())

            # Execute AI Scripts
            with st.spinner("🧠 AI Engines Analyzing Data..."):
                create_combined_dataset()
                train_hybrid_model()

            # Display Results
            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            if os.path.exists(res_path):
                df = pd.read_csv(res_path)
                
                st.subheader("📊 Sprint Risk Dashboard")
                cols = st.columns(len(df))
                
                for i, (_, row) in enumerate(df.iterrows()):
                    with cols[i]:
                        risk = str(row['risk_level']).strip().upper()
                        # Color logic
                        if "HIGH" in risk:
                            icon = "🔴"
                        elif "MEDIUM" in risk:
                            icon = "🟠"
                        else:
                            icon = "🟢"
                        st.metric(label=f"Sprint {row['sprint']}", value=f"{icon} {risk}")

                st.markdown("---")
                st.subheader("📈 Risk Trend Analysis")
                # Labeled Chart
                chart_data = df[['ambiguity_score', 'overload_score']]
                st.line_chart(chart_data)
                st.caption("Legend: Dark Blue = Ambiguity Score | Light Blue = Overload Score")

                # Downloadable Report
                st.markdown("---")
                report_path = os.path.join(root_path, "results", "ambiguity_report.csv")
                if os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button("📥 Download Technical Risk Report", f, "risk_report.csv", "text/csv")
            else:
                st.error("Analysis completed but result file not found.")

        except Exception as e:
            st.error(f"Error during analysis: {e}")
else:
    st.warning("👈 Please upload both Requirements and Sprint files in the sidebar.")