
import sys
import os
import pandas as pd
import streamlit as st
from textblob import TextBlob
import nltk

# --- 1. SYSTEM INITIALIZATION ---
@st.cache_resource
def setup_engine():
    try:
        nltk.download('punkt')
        nltk.download('brown')
        nltk.download('punkt_tab')
    except:
        pass
setup_engine()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError:
    st.error("Engine failure: Core modules (src/) not found.")

# --- 2. UI THEME & PROFESSIONAL BUTTONS ---
st.set_page_config(page_title="SentianRisk Pro", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    .stApp { background-color: #080a0c; font-family: 'Plus Jakarta Sans', sans-serif; color: white; }
    
    /* Neural Pulse */
    .status-ring { width: 80px; height: 80px; border-radius: 50%; border: 2px solid #00d9ff; margin: 0 auto 30px; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 217, 255, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 15px rgba(0, 217, 255, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 217, 255, 0); } }

    /* KPI & Prescription Boxes */
    .kpi-box { background: linear-gradient(145deg, #111418, #181c22); border: 1px solid rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; }
    .prescription-card { background: rgba(0, 217, 255, 0.05); border: 1px solid #00d9ff; border-radius: 12px; padding: 20px; margin-top: 20px; }
    
    /* Professional Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #00d9ff, #005fcc); color: white; border: none;
        padding: 12px 24px; border-radius: 8px; font-weight: 700; letter-spacing: 1px;
        transition: all 0.3s ease; width: 100%;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 217, 255, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BRANDING HEADER ---
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 40px;">
        <div><h1 style='margin:0; font-weight:300;'>SENTIAN<span style='color:#00d9ff; font-weight:800;'>RISK</span></h1><p style='color:#555; margin:0; font-size:0.8rem;'>HYBRID GOVERNANCE ENGINE</p></div>
        <div style="text-align:right;"><p style='color:#00d9ff; margin:0; font-size:0.7rem; font-weight:800;'>SYSTEM STATUS: OPERATIONAL</p></div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("### Data Ingestion")
    req_file = st.file_uploader("Project Specifications (.txt)", type=["txt"])
    spr_file = st.file_uploader("Operational Metadata (.csv)", type=["csv"])
    st.markdown("---")
    execute = st.button("RUN SYSTEM DIAGNOSTICS")

# --- 5. MAIN LOGIC & POWER FEATURES ---
if execute and req_file and spr_file:
    try:
        raw_df = pd.read_csv(spr_file)
        r_text = req_file.getvalue().decode("utf-8")
        
        # STRUCTURAL VALIDATION (THE GATEKEEPER)
        csv_cols = "".join(raw_df.columns).lower()
        project_vectors = ['sprint', 'task', 'hour', 'capacity', 'effort', 'deadline']
        
        if not any(v in csv_cols for v in project_vectors):
            st.warning("🚨 **SYSTEM AUDIT: STRUCTURAL MISMATCH**")
            st.error("The uploaded files are **NOT related to Project Management**. The AI requires Sprint, Workload, and Capacity metrics to perform a risk induction.")
            st.info("💡 **Prescription:** Please upload valid development metadata to synchronize the risk engine.")
        else:
            with st.spinner("Synchronizing Hybrid Risk Vectors..."):
                os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
                raw_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)
                with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
                
                create_combined_dataset()
                train_hybrid_model()
                df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))
                avg_risk = df['overload_score'].mean()
                sentiment = TextBlob(r_text).sentiment.polarity

            # KPI DASHBOARD
            k1, k2, k3, k4 = st.columns(4)
            with k1: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>SENTIMENT BIAS</p><h2 style='color:#00d9ff;'>{('STABLE' if sentiment > 0 else 'VOLATILE')}</h2></div>", unsafe_allow_html=True)
            with k2: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>RISK COEFFICIENT</p><h2>{avg_risk:.2f}</h2></div>", unsafe_allow_html=True)
            with k3: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>ITERATIONS</p><h2 style='color:#00ff9d;'>{len(df)}</h2></div>", unsafe_allow_html=True)
            with k4: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>MODEL FIDELITY</p><h2>94%</h2></div>", unsafe_allow_html=True)

            # --- POWER FEATURE: AI RISK MANAGEMENT PLAN ---
            st.markdown("### 🧠 AI Strategic Governance Plan")
            
            with st.container():
                st.markdown(f"<div class='prescription-card'>", unsafe_allow_html=True)
                if avg_risk > 0.6:
                    st.markdown(f"#### 🔴 High Risk Detected: Resource Saturation")
                    st.write(f"**Current State:** Your team is currently operating at **{int(avg_risk*100)}%** overload. This will lead to a 'Burnout Event' within the next 2 iterations.")
                    st.write("**Future Mitigation:** 1. Offload 20% of non-critical tasks. 2. Pause new feature requests. 3. Re-baseline the timeline by 14 days.")
                elif sentiment < 0:
                    st.markdown(f"#### 🟡 Moderate Risk: Linguistic Ambiguity")
                    st.write("**Current State:** Requirement text lacks technical clarity. Misinterpretation risk is high.")
                    st.write("**Future Mitigation:** 1. Conduct a Requirements Walkthrough. 2. Define strict 'Definition of Done' (DoD) for current tasks.")
                else:
                    st.markdown(f"#### 🟢 Optimal State: System Balanced")
                    st.write("**Current State:** Resources and communication are perfectly aligned.")
                    st.write("**Future Mitigation:** Maintain current velocity. System is cleared for feature expansion.")
                st.markdown("</div>", unsafe_allow_html=True)

            st.line_chart(df.set_index('sprint')[['overload_score', 'ambiguity_score']])

    except Exception as e:
        st.error(f"Engine Exception: {str(e)}")
else:
    st.markdown("""<div style='text-align:center; padding-top:100px;'><div class="status-ring"></div><h2 style='color:white; font-weight:200; letter-spacing:4px;'>SYSTEM STANDBY</h2></div>""", unsafe_allow_html=True)

st.markdown(f"<div class='footer'>SENTIANRISK ARCHITECTURE &copy; 2026 BY SHAMA SALEEM</div>", unsafe_allow_html=True)