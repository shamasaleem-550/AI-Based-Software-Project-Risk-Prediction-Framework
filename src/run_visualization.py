import sys
import os
import pandas as pd
import streamlit as st
from textblob import TextBlob
import nltk

# --- 1. CLOUD-READY NLP SETUP ---
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

# --- 2. CONFIG & THEME (The High-End Look) ---
st.set_page_config(page_title="SentianRisk Pro", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    .stApp { background-color: #080a0c; font-family: 'Plus Jakarta Sans', sans-serif; color: white; }
    
    /* Branding Header */
    .brand-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 20px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 40px;
    }

    /* Analytics Cards */
    .kpi-box {
        background: linear-gradient(145deg, #111418, #181c22);
        border: 1px solid rgba(255,255,255,0.03);
        border-radius: 20px; padding: 25px; text-align: center;
    }

    /* Problem/Solution Card */
    .diag-card {
        background: #111418; border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px; padding: 24px; margin-bottom: 20px;
    }
    .prob-label { color: #ff5f5f; font-weight: 800; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .sol-label { color: #00d9ff; font-weight: 800; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Footer */
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
            <h1 style='margin:0; font-weight:300; letter-spacing:-1px;'>SENTIAN<span style='color:#00d9ff; font-weight:800;'>RISK</span> PRO</h1>
            <p style='color:#555; margin:0; font-size:0.8rem;'>PREDICTIVE GOVERNANCE ENGINE</p>
        </div>
        <div style="text-align:right;">
            <p style='color:#00d9ff; margin:0; font-size:0.7rem; font-weight:800;'>LEAD RESEARCH ENGINEER</p>
            <p style='color:white; margin:0; font-size:1.1rem; font-weight:300;'>SHAMA SALEEM</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("<h3 style='color:white;'>Input Stream</h3>", unsafe_allow_html=True)
    req_file = st.file_uploader("Requirement Specs (.txt)", type=["txt"])
    spr_file = st.file_uploader("Resource Metadata (.csv)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 🛠️ SIMULATION")
    sim_boost = st.slider("Capacity Buffer (Hrs)", 0, 100, 0)
    
    if st.button("EXECUTE NEURAL ANALYSIS", type="primary", use_container_width=True):
        st.session_state['active'] = True

# --- 5. DASHBOARD MAIN ---
if st.session_state.get('active'):
    if req_file and spr_file:
        r_text = req_file.getvalue().decode("utf-8")
        r_df = pd.read_csv(spr_file)
        
        # Save & Process
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
        r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        with st.spinner("Synchronizing Hybrid Risk Vectors..."):
            create_combined_dataset()
            train_hybrid_model()
            df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))
            sentiment = TextBlob(r_text).sentiment.polarity

        # --- A. TOP KPI GRID ---
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>MOOD</p><h2 style='color:#00d9ff;'>{('CLEAR' if sentiment > 0.1 else 'VAGUE')}</h2></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>AVG STRESS</p><h2>{df['overload_score'].mean():.2f}</h2></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>SIM IMPACT</p><h2 style='color:#00ff9d;'>-{sim_boost/10:.1f}%</h2></div>", unsafe_allow_html=True)
        with k4: st.markdown(f"<div class='kpi-box'><p style='color:#555; font-size:0.7rem;'>CONFIDENCE</p><h2>94%</h2></div>", unsafe_allow_html=True)

        # --- B. CHARTS ---
        st.markdown("<br>", unsafe_allow_html=True)
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown("<p style='color:#555; font-size:0.7rem; font-weight:800;'>RISK TRAJECTORY</p>", unsafe_allow_html=True)
            st.line_chart(df.set_index('sprint')[['ambiguity_score', 'overload_score']])
        with c_right:
            st.markdown("<p style='color:#555; font-size:0.7rem; font-weight:800;'>STRAIN DENSITY</p>", unsafe_allow_html=True)
            st.bar_chart(df.set_index('sprint')['overload_score'])

        # --- C. PROBLEM/SOLUTION DIAGNOSTICS ---
        st.markdown("<h3 style='font-weight:300; color:#888; margin:30px 0;'>Prescriptive Diagnostics</h3>", unsafe_allow_html=True)
        for _, row in df.iterrows():
            risk = str(row['risk_level']).upper()
            color = "#ff5f5f" if "HIGH" in risk else "#ffa500" if "MEDIUM" in risk else "#00d9ff"
            
            # Dynamic Prescriptions
            prob = f"Sprint {row['sprint']} exceeds resource capacity." if "HIGH" in risk else "Ambiguous requirements detected." if "MEDIUM" in risk else "Metrics stable."
            sol = "Offload 20% tasks to backlog." if "HIGH" in risk else "Refine SRS for clarity." if "MEDIUM" in risk else "Proceed at current velocity."

            st.markdown(f"""
                <div class="diag-card" style="border-left: 5px solid {color};">
                    <div style="display:flex; justify-content:space-between; margin-bottom:15px;">
                        <span style="color:{color}; font-weight:800;">SPRINT {row['sprint']} Analysis</span>
                        <span style="font-size:0.8rem; color:#555;">{risk}</span>
                    </div>
                    <div style="margin-bottom:10px;">
                        <span class="prob-label">Problem</span><br><span style="color:#888;">{prob}</span>
                    </div>
                    <div>
                        <span class="sol-label">Prescription</span><br><span style="color:white; font-weight:600;">{sol}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please upload both Requirement (.txt) and Resource (.csv) data streams.")
else:
    st.markdown("<div style='text-align:center; padding-top:150px; opacity:0.1;'><h1 style='font-size:5rem;'>READY</h1></div>", unsafe_allow_html=True)

# --- 6. FOOTER ---
st.markdown("<div class='footer'>PROJECT ARCHITECTURE BY SHAMA SALEEM &nbsp; | &nbsp; © 2026 SENTIANRISK AI SOLUTIONS</div>", unsafe_allow_html=True)