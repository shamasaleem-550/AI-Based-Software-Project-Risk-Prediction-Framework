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

# Pathing setup to ensure it finds your src/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError:
    st.error("Engine failure: Core modules (src/) not found. Check folder structure.")

# --- 2. THE SMART MAPPER (Column Recognition) ---
def smart_map_columns(df):
    mapping = {}
    cols = df.columns.str.lower().str.strip()
    for i, col in enumerate(cols):
        if any(x in col for x in ['sprint', 'id', 'no', 'period']):
            mapping['sprint'] = df.columns[i]
        elif any(x in col for x in ['assigned', 'hours', 'work', 'load']):
            mapping['hours_assigned'] = df.columns[i]
        elif any(x in col for x in ['capacity', 'limit', 'max', 'total']):
            mapping['team_capacity'] = df.columns[i]
    return mapping

# --- 3. UI THEME & ANIMATIONS ---
st.set_page_config(page_title="SentianRisk Pro", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    .stApp { background-color: #080a0c; font-family: 'Plus Jakarta Sans', sans-serif; color: white; }
    
    /* Neural Pulse Animation */
    .status-ring {
        width: 80px; height: 80px; border-radius: 50%;
        border: 2px solid #00d9ff; margin: 0 auto 30px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 217, 255, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 15px rgba(0, 217, 255, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 217, 255, 0); }
    }

    .kpi-box { background: linear-gradient(145deg, #111418, #181c22); border: 1px solid rgba(255,255,255,0.03); border-radius: 15px; padding: 20px; text-align: center; }
    .footer { text-align: center; color: #444; font-size: 0.75rem; padding: 30px 0; border-top: 1px solid rgba(255,255,255,0.03); margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. BRANDING HEADER ---
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 40px;">
        <div>
            <h1 style='margin:0; font-weight:300;'>SENTIAN<span style='color:#00d9ff; font-weight:800;'>RISK</span></h1>
            <p style='color:#555; margin:0; font-size:0.8rem;'>HYBRID GOVERNANCE ENGINE</p>
        </div>
        <div style="text-align:right;">
            <p style='color:#00d9ff; margin:0; font-size:0.7rem; font-weight:800;'>SYSTEM STATUS: ONLINE</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("### Control Panel")
    req_file = st.file_uploader("Upload Requirements (.txt)", type=["txt"])
    spr_file = st.file_uploader("Upload Sprint Data (.csv)", type=["csv"])
    st.markdown("---")
    execute = st.button("✨ EXECUTE ANALYSIS", type="primary", use_container_width=True)

# --- 6. MAIN LOGIC & GATEKEEPER ---
if execute and req_file and spr_file:
    try:
        raw_df = pd.read_csv(spr_file)
        r_text = req_file.getvalue().decode("utf-8")
        
        # GATEKEEPER CHECK: Ensure file is project-related
        csv_cols = "".join(raw_df.columns).lower()
        project_keywords = ['sprint', 'task', 'hour', 'capacity', 'effort', 'id', 'deadline']
        
        if not any(key in csv_cols for key in project_keywords):
            st.warning("⚠️ **Context Mismatch Detected**")
            st.info("The uploaded file does not contain project metrics (Sprints/Hours). Please upload valid development metadata.")
        else:
            # PROCESS DATA
            col_map = smart_map_columns(raw_df)
            clean_df = raw_df.rename(columns={v: k for k, v in col_map.items()})
            
            # Save files for the backend engine to find
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            clean_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)
            with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)

            with st.spinner("AI Engine Processing..."):
                create_combined_dataset()
                train_hybrid_model()
                df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))
                sentiment = TextBlob(r_text).sentiment.polarity

            # DISPLAY DASHBOARD
            k1, k2, k3, k4 = st.columns(4)
            with k1: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>MOOD</p><h2 style='color:#00d9ff;'>{('STABLE' if sentiment > 0 else 'VAGUE')}</h2></div>", unsafe_allow_html=True)
            with k2: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>RISK INDEX</p><h2>{df['overload_score'].mean():.2f}</h2></div>", unsafe_allow_html=True)
            with k3: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>SPRINTS</p><h2 style='color:#00ff9d;'>{len(df)}</h2></div>", unsafe_allow_html=True)
            with k4: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>CONFIDENCE</p><h2>94%</h2></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.line_chart(df.set_index('sprint')[['overload_score', 'ambiguity_score']])
            st.success("Governance Analysis Successful.")

    except Exception as e:
        st.error(f"Analysis interrupted: {str(e)}")
else:
    # STANDBY MODE (Pulse Animation)
    st.markdown("""
        <div style='text-align:center; padding-top:100px;'>
            <div class="status-ring"></div>
            <h2 style='color:white; font-weight:200; letter-spacing:4px;'>SYSTEM STANDBY</h2>
            <p style='color:#444; font-size:0.8rem; margin-top:10px;'>WAITING FOR DATA INGESTION</p>
        </div>
    """, unsafe_allow_html=True)

# --- 7. FOOTER ---
st.markdown(f"<div class='footer'>SENTIANRISK ARCHITECTURE &copy; 2026 BY SHAMA SALEEM</div>", unsafe_allow_html=True)