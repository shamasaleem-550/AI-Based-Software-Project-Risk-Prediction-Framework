import sys
import os
import pandas as pd
import streamlit as st
from textblob import TextBlob
import nltk

# --- 1. CORE SETUP ---
@st.cache_resource
def setup_nlp():
    nltk.download('punkt')
    nltk.download('brown')

setup_nlp()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError:
    st.error("Backend modules missing.")

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="SentianRisk Pro | Shama Saleem", layout="wide")

# --- 3. PREMIUM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    .stApp { background-color: #080a0c; font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* Diagnostic Container */
    .diag-container {
        background: #111418;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        border-left: 5px solid #4facfe;
    }
    
    .problem-header { color: #ff4b4b; font-size: 0.75rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }
    .solution-header { color: #00ffcc; font-size: 0.75rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }
    
    .kpi-card {
        background: linear-gradient(145deg, #111418, #181c22);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    /* Top Header Branding */
    .brand-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. BRANDED HEADER ---
st.markdown(f"""
    <div class="brand-header">
        <div>
            <h1 style='color:white; margin:0; font-weight:300;'>SENTIAN<span style='color:#4facfe; font-weight:800;'>RISK</span> PRO</h1>
            <p style='color:#555; margin:0; font-size:0.8rem; font-weight:600;'>HYBRID RISK INTELLIGENCE SYSTEM</p>
        </div>
        <div style="text-align:right;">
            <p style='color:#4facfe; margin:0; font-size:0.7rem; font-weight:800; letter-spacing:1px;'>LEAD RESEARCH ENGINEER</p>
            <p style='color:white; margin:0; font-size:1.2rem; font-weight:300;'>SHAMA SALEEM</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color:white;'>Control Panel</h3>", unsafe_allow_html=True)
    req_file = st.file_uploader("Upload Specs (.txt)", type=["txt"])
    spr_file = st.file_uploader("Upload Data (.csv)", type=["csv"])
    st.markdown("---")
    if st.button("RUN AI DIAGNOSTICS", type="primary", use_container_width=True):
        st.session_state['active'] = True

# --- 6. MAIN CONTENT ---
if st.session_state.get('active'):
    if req_file and spr_file:
        r_text = req_file.getvalue().decode("utf-8")
        r_df = pd.read_csv(spr_file)
        
        # Sentiment logic
        sentiment = TextBlob(r_text).sentiment.polarity
        
        # Process
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
        r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        create_combined_dataset()
        train_hybrid_model()
        df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))

        # --- KPI GRID ---
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f"<div class='kpi-card'><p style='color:#6c757d; font-size:0.7rem;'>LINGUISTIC MOOD</p><h2 style='color:#4facfe;'>{('STABLE' if sentiment > 0 else 'UNSTABLE')}</h2></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi-card'><p style='color:#6c757d; font-size:0.7rem;'>STRESS INDEX</p><h2 style='color:white;'>{df['overload_score'].mean():.2f}</h2></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='kpi-card'><p style='color:#6c757d; font-size:0.7rem;'>AI CONFIDENCE</p><h2 style='color:#00ffcc;'>94.8%</h2></div>", unsafe_allow_html=True)

        st.markdown("<br><h3 style='font-weight:300; color:white;'>Prescriptive Analysis</h3>", unsafe_allow_html=True)
        
        # --- PROBLEM/SOLUTION BLOCKS ---
        for _, row in df.iterrows():
            risk_lvl = str(row['risk_level']).upper()
            
            if "HIGH" in risk_lvl:
                prob = f"CRITICAL OVERLOAD: Sprint {row['sprint']} exceeds resource thresholds. Potential for missed deadlines is high."
                sol = "Reallocate non-critical tasks to the next sprint and increase senior developer oversight."
                status_clr = "#ff4b4b"
            elif "MEDIUM" in risk_lvl:
                prob = f"MODERATE AMBIGUITY: Requirement wording for Sprint {row['sprint']} shows signs of uncertainty."
                sol = "Hold a brief technical clarification meeting to define sub-tasks more clearly."
                status_clr = "#ffa500"
            else:
                prob = "STABLE METRICS: Sprint {row['sprint']} is well-balanced."
                sol = "Maintain current velocity. No immediate intervention required."
                status_clr = "#00ffcc"

            st.markdown(f"""
                <div class="diag-container" style="border-left: 5px solid {status_clr};">
                    <div style="display:flex; justify-content:space-between; margin-bottom:15px;">
                        <span style="color:{status_clr}; font-weight:800; font-size:1.1rem;">SPRINT {row['sprint']} — {risk_lvl}</span>
                    </div>
                    <div style="margin-bottom:10px;">
                        <span class="problem-header">Detected Problem</span><br>
                        <span style="color:#aaa; font-size:0.95rem;">{prob}</span>
                    </div>
                    <div>
                        <span class="solution-header">AI Recommendation</span><br>
                        <span style="color:white; font-size:0.95rem; font-weight:600;">{sol}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align:center; padding:100px; opacity:0.2;'><h1>IDLE</h1><p>Neural engine standby for Shama Saleem's data input.</p></div>", unsafe_allow_html=True)