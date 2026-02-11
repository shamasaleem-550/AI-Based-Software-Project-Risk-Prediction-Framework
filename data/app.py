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
st.set_page_config(page_title="SentianRisk AI", layout="wide", page_icon="🛡️")

# Professional UI Styling
st.markdown("""
    <style>
    .stMetric { 
        background-color: #1E1E1E !important; 
        color: white !important;
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #3E3E3E;
    }
    .explanation-box {
        background-color: #262730;
        border-left: 5px solid #FF4B4B;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ SentianRisk AI: Intelligent Project Oversight")
st.markdown(f"**Developed by SHAMA SALEEM** | FYP Edition: Hybrid NLP & Resource Analysis")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📁 Data Input")
    requirements_file = st.file_uploader("1. Requirements (.txt)", type=["txt"])
    sprint_file = st.file_uploader("2. Sprint Tasks (.csv)", type=["csv"])
    
    st.markdown("---")
    st.subheader("💡 FYP Research Templates")
    sample_req = "System must allow user login. The interface should be fast and scalable."
    st.download_button("📥 Sample Requirements", sample_req, "sample_req.txt")
    sample_csv = "sprint,task_name,hours_assigned,developer_capacity\n1,UI Design,45,40\n2,Database,20,40\n3,API Integration,60,40"
    st.download_button("📥 Sample Sprint Data", sample_csv, "sample_sprint.csv")

# --- 5. MAIN LOGIC ---
if requirements_file and sprint_file:
    if st.button("🔍 Run Advanced AI Risk Analysis", use_container_width=True):
        try:
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(root_path, "results"), exist_ok=True)

            with open(os.path.join(root_path, "data", "requirements.txt"), "wb") as f:
                f.write(requirements_file.getvalue())
            with open(os.path.join(root_path, "data", "sprint_tasks.csv"), "wb") as f:
                f.write(sprint_file.getvalue())

            with st.spinner("🧠 Deep Learning Models Analyzing Requirements..."):
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
                        # Traffic Light Logic
                        if "HIGH" in risk: icon, color = "🔴", "#FF4B4B"
                        elif "MEDIUM" in risk: icon, color = "🟠", "#FFA500"
                        else: icon, color = "🟢", "#00FF00"
                        
                        st.metric(label=f"Sprint {row['sprint']}", value=f"{icon} {risk}")
                        
                        # --- MODIFICATION: AI EXPLANATION LOGIC ---
                        with st.expander("ℹ️ AI Risk Reason"):
                            if "HIGH" in risk:
                                st.write(f"⚠️ **Reasoning:** Hours assigned ({row['hours_assigned']}) significantly exceed capacity ({row['developer_capacity']}). High potential for burnout.")
                            elif "MEDIUM" in risk:
                                st.write(f"🟠 **Reasoning:** Ambiguity score is moderate. Requirements may need more detail to avoid rework.")
                            else:
                                st.write("✅ **Reasoning:** Resources are balanced and requirements are clear.")

                st.markdown("---")
                # --- MODIFICATION: TREND ANALYSIS WITH LABELS ---
                st.subheader("📈 Quantitative Risk Trends")
                st.line_chart(df[['ambiguity_score', 'overload_score']])
                st.info("The Blue line represents Linguistic Ambiguity (NLP), and the Light Blue represents Resource Strain.")
                
            else:
                st.error("Analysis completed but results were not generated.")

        except Exception as e:
            st.error(f"Critical Error: {e}")
else:
    st.warning("👈 Please upload files to start the FYP Analysis Framework.")