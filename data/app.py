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
    st.error(f"Module import failed. Error: {e}")

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Project Risk Predictor", layout="wide")

# Custom CSS for better visibility of the "White Boxes"
st.markdown("""
    <style>
    .stMetric { 
        background-color: #1E1E1E !important; 
        color: white !important;
        padding: 20px; 
        border-radius: 12px; 
        border: 2px solid #4B4B4B;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    [data-testid="stMetricValue"] {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI-Based Software Project Risk Predictor")
st.markdown("Developed by **SHAMA SALEEM**")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📁 Data Input")
    requirements_file = st.file_uploader("1. Upload Requirements (.txt)", type=["txt"])
    sprint_file = st.file_uploader("2. Upload Sprint Tasks (.csv)", type=["csv"])
    st.markdown("---")
    st.subheader("💡 Sample Data")
    sample_req = "System must be fast. Requirements are flexible."
    st.download_button("📥 Sample .txt", sample_req, "sample_req.txt")
    sample_csv = "sprint,task_name,hours_assigned,developer_capacity\n1,Task A,45,40\n2,Task B,20,40"
    st.download_button("📥 Sample .csv", sample_csv, "sample_sprint.csv")

# --- 5. MAIN LOGIC ---
if requirements_file and sprint_file:
    if st.button("🔍 Run Full AI Risk Analysis", use_container_width=True):
        try:
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(root_path, "results"), exist_ok=True)

            with open(os.path.join(root_path, "data", "requirements.txt"), "wb") as f:
                f.write(requirements_file.getvalue())
            with open(os.path.join(root_path, "data", "sprint_tasks.csv"), "wb") as f:
                f.write(sprint_file.getvalue())

            with st.spinner("🧠 AI Processing..."):
                create_combined_dataset()
                train_hybrid_model()

            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            if os.path.exists(res_path):
                df = pd.read_csv(res_path)
                st.subheader("📊 Sprint Risk Dashboard")
                cols = st.columns(len(df))
                
                for i, (_, row) in enumerate(df.iterrows()):
                    with cols[i]:
                        risk = str(row['risk_level']).strip().upper()
                        # FORCE COLOR ICON BASED ON TEXT
                        if "HIGH" in risk:
                            icon = "🔴"
                        elif "MEDIUM" in risk:
                            icon = "🟠"
                        else:
                            icon = "🟢"
                        st.metric(label=f"Sprint {row['sprint']}", value=f"{icon} {risk}")

                st.markdown("---")
                st.write("### 📈 Risk Trends")
                st.line_chart(df[['ambiguity_score', 'overload_score']])
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.warning("👈 Please upload files in the sidebar.")