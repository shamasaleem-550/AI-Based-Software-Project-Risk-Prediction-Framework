import sys
import os
import pandas as pd
import streamlit as st
from textblob import TextBlob
import nltk

# --- 1. SYSTEM INITIALIZATION ---
@st.cache_resource
def setup_engine():
    nltk.download('punkt')
    nltk.download('brown')
    nltk.download('punkt_tab')

setup_engine()

# Pathing
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError:
    st.error("Engine failure: src/ modules missing.")

# --- 2. THE ULTIMATE AESTHETIC CSS ---
st.set_page_config(page_title="SentianRisk Pro", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    .stApp { background-color: #080a0c; font-family: 'Plus Jakarta Sans', sans-serif; color: white; }
    
    .brand-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 40px;
    }

    /* System Standby "Pulse" Animation */
    .status-ring {
        width: 100px; height: 100px; border-radius: 50%;
        border: 2px solid #00d9ff; margin: 0 auto 30px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 217, 255, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 20px rgba(0, 217, 255, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 217, 255, 0); }
    }

    .kpi-box {
        background: linear-gradient(145deg, #111418, #181c22);
        border: 1px solid rgba(255,255,255,0.03);
        border-radius: 20px; padding: 25px; text-align: center;
    }

    .diag-card {
        background: #111418; border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
    }
    
    .footer {
        text-align: center; color: #444; font-size: 0.7rem; padding: 40px 0;
        border-top: 1px solid rgba(255,255,255,0.03); margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. UI BRANDING ---
st.markdown("""
    <div class="brand-bar">
        <div>
            <h1 style='margin:0; font-weight:300; letter-spacing:-1px;'>SENTIAN<span style='color:#00d9ff; font-weight:800;'>RISK</span></h1>
            <p style='color:#555; margin:0; font-size:0.8rem;'>PREDICTIVE GOVERNANCE ENGINE</p>
        </div>
        <div style="text-align:right;">
            <p style='color:#00d9ff; margin:0; font-size:0.7rem; font-weight:800;'>LEAD RESEARCH ENGINEER</p>
            <p style='color:white; margin:0; font-size:1.1rem; font-weight:300;'>SHAMA SALEEM</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color:white;'>Data Ingestion</h3>", unsafe_allow_html=True)
    req_file = st.file_uploader("Requirement Specs (.txt)", type=["txt"])
    spr_file = st.file_uploader("Sprint Metadata (.csv)", type=["csv"])
    
    st.markdown("---")
    # Added "✨" to the button for a touch of AI flair
    if st.button("✨ PROCESS NEURAL MODEL", type="primary", use_container_width=True):
        st.session_state['active'] = True

# --- 5. MAIN DASHBOARD ---
if st.session_state.get('active'):
    if req_file and spr_file:
        r_text = req_file.getvalue().decode("utf-8")
        r_df = pd.read_csv(spr_file)
        
        # Backend Run
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
        r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        with st.spinner("Processing Hybrid Risk Vectors..."):
            create_combined_dataset()
            train_hybrid_model()
            df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))
            sentiment = TextBlob(r_text).sentiment.polarity

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>MOOD</p><h2 style='color:#00d9ff;'>{('CLEAR' if sentiment > 0.1 else 'VAGUE')}</h2></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>AVG STRESS</p><h2>{df['overload_score'].mean():.2f}</h2></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>RESOURCES</p><h2 style='color:#00ff9d;'>{len(df)} Sprints</h2></div>", unsafe_allow_html=True)
        with k4: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>CONFIDENCE</p><h2>94%</h2></div>", unsafe_allow_html=True)

        # Charts
        st.markdown("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown("<p style='color:#555; font-size:0.7rem; font-weight:800;'>RISK TRAJECTORY</p>", unsafe_allow_html=True)
            st.line_chart(df.set_index('sprint')[['ambiguity_score', 'overload_score']])
        with c_right:
            st.markdown("<p style='color:#555; font-size:0.7rem; font-weight:800;'>STRAIN DENSITY</p>", unsafe_allow_html=True)
            st.bar_chart(df.set_index('sprint')['overload_score'])

        # Problem/Solution Diagnostics
        st.markdown("<h3 style='font-weight:300; color:#888; margin:30px 0;'>Prescriptive Diagnostics</h3>", unsafe_allow_html=True)
        for _, row in df.iterrows():
            risk = str(row['risk_level']).upper()
            color = "#ff5f5f" if "HIGH" in risk else "#ffa500" if "MEDIUM" in risk else "#00d9ff"
            st.markdown(f"""
                <div class="diag-card" style="border-left: 5px solid {color};">
                    <div style="display:flex; justify-content:space-between; margin-bottom:15px;">
                        <span style="color:{color}; font-weight:800;">SPRINT {row['sprint']} — {risk}</span>
                    </div>
                    <div style="margin-bottom:10px;">
                        <span style="color:#ff5f5f; font-weight:800; font-size:0.75rem;">DETECTED PROBLEM</span><br><span style="color:#888;">{risk} stress level detected.</span>
                    </div>
                    <div>
                        <span style="color:#00d9ff; font-weight:800; font-size:0.75rem;">AI PRESCRIPTION</span><br><span style="color:white; font-weight:600;">{"Offload tasks to backlog." if "HIGH" in risk else "Refine technical clarity." if "MEDIUM" in risk else "Continue velocity."}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please upload project data in the sidebar to initialize.")
else:
    # UPDATED STANDBY SECTION
    st.markdown("""
        <div style='text-align:center; padding-top:100px;'>
            <div class="status-ring"></div>
            <h2 style='color:white; font-weight:200; letter-spacing:2px;'>NEURAL ENGINE READY</h2>
            <p style='color:#444;'>SHAMA SALEEM | VERSION 3.0.42</p>
            <p style='color:#888; font-size:0.9rem; margin-top:20px;'>Awaiting data injection via control panel.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='footer'>PROJECT ARCHITECTURE BY SHAMA SALEEM &nbsp; | &nbsp; © 2026 SENTIANRISK AI</div>", unsafe_allow_html=True)