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
    st.error(f"Module import failed. Check structure. Error: {e}")

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="SentianRisk AI", layout="wide", page_icon="🛡️")

# UI Styling for Professional Dark Mode Cards
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
        font-weight: bold;
    }
    .main { background-color: #0E1117; }
    </style>
    """, unsafe_allow_html=True)

# THE NEW PROFESSIONAL TITLE
st.title("🛡️ SentianRisk AI: Intelligent Project Oversight")
st.markdown(f"**Developed by SHAMA SALEEM** | Hybrid NLP & Resource Analytics Framework")

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📁 Data Input")
    st.write("Upload project files for AI processing.")
    
    requirements_file = st.file_uploader("1. Requirements (.txt)", type=["txt"])
    sprint_file = st.file_uploader("2. Sprint Tasks (.csv)", type=["csv"])
    
    st.markdown("---")
    st.subheader("💡 Demo Templates")
    
    # Requirements Sample
    sample_req = "The system must be fast. We require robust security and flexible login."
    st.download_button("📥 Download Sample .txt", sample_req, "sample_req.txt")

    # Sprint Sample
    sample_csv = "sprint,task_name,hours_assigned,developer_capacity\n1,Module A,45,40\n2,Module B,22,40\n3,Module C,58,40"
    st.download_button("📥 Download Sample .csv", sample_csv, "sample_sprint.csv")
    
    st.markdown("---")
    st.caption("AI detects ambiguity in text and capacity overloads.")

# --- 5. MAIN LOGIC ---
if requirements_file and sprint_file:
    if st.button("🔍 Run Intelligent Risk Analysis", use_container_width=True):
        try:
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(root_path, "results"), exist_ok=True)

            # Save inputs
            with open(os.path.join(root_path, "data", "requirements.txt"), "wb") as f:
                f.write(requirements_file.getvalue())
            with open(os.path.join(root_path, "data", "sprint_tasks.csv"), "wb") as f:
                f.write(sprint_file.getvalue())

            # Run Backend
            with st.spinner("🧠 Processing Linguistic and Resource Metrics..."):
                create_combined_dataset()
                train_hybrid_model()

            # Result Display
            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            if os.path.exists(res_path):
                df = pd.read_csv(res_path)
                st.subheader("📊 Sprint Risk Dashboard")
                cols = st.columns(len(df))
                
                for i, (_, row) in enumerate(df.iterrows()):
                    with cols[i]:
                        risk = str(row['risk_level']).strip().upper()
                        # Dynamic Color Icons
                        if "HIGH" in risk:
                            icon = "🔴"
                        elif "MEDIUM" in risk:
                            icon = "🟠"
                        else:
                            icon = "🟢"
                        st.metric(label=f"Sprint {row['sprint']}", value=f"{icon} {risk}")

                st.markdown("---")
                st.subheader("📈 Technical Risk Trends")
                # Visualizing the hybrid scores
                st.line_chart(df[['ambiguity_score', 'overload_score']])
                st.info("💡 Ambiguity (Blue) vs Resource Overload (Light Blue)")
            else:
                st.error("Error: Result file not generated.")

        except Exception as e:
            st.error(f"Critical Error: {e}")
else:
    st.warning("👈 Please upload project data in the sidebar to begin analysis.")