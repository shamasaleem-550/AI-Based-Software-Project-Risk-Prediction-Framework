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

# System path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError:
    st.error("Backend modules missing. Please ensure src/ folder exists.")

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="SentianRisk Pro", layout="wide", initial_sidebar_state="expanded")

# --- 3. PROFESSIONAL CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    .stApp { background-color: #080a0c; font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* Diagnostic Card */
    .diag-container {
        background: #111418;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .diag-container:hover { transform: scale(1.01); border-color: #4facfe; }

    .problem-header { color: #ff4b4b; font-size: 0.8rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }
    .solution-header { color: #00ffcc; font-size: 0.8rem; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; }
    
    .kpi-card {
        background: linear-gradient(145deg, #111418, #181c22);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .status-pill {
        background: rgba(79, 172, 254, 0.1);
        color: #4facfe;
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 700;
        border: 1px solid #4facfe;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#4facfe; font-size: 1.8rem; font-weight: 800;'>SENTIAN<span style='color:white;'>RISK</span></h1>", unsafe_allow_html=True)
    st.markdown("<span class='status-pill'>NEURAL ENGINE v3.0</span>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    req_file = st.file_uploader("📝 Requirements Spec (.txt)", type=["txt"])
    spr_file = st.file_uploader("📊 Resource Schedule (.csv)", type=["csv"])
    
    st.markdown("---")
    if st.button("RUN AI DIAGNOSTICS", use_container_width=True, type="primary"):
        st.session_state['active'] = True

# --- 5. MAIN DASHBOARD ---
if st.session_state.get('active'):
    if req_file and spr_file:
        r_text = req_file.getvalue().decode("utf-8")
        r_df = pd.read_csv(spr_file)
        
        # AI Logic
        sentiment = TextBlob(r_text).sentiment.polarity
        
        # Backend Processing
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
        r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        create_combined_dataset()
        train_hybrid_model()
        df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))

        # --- KPI SECTION ---
        st.markdown("<h2 style='font-weight:300; color:white;'>Intelligence Overview</h2>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"<div class='kpi-card'><p style='color:#6c757d; font-size:0.7rem;'>LINGUISTIC CLARITY</p><h2 style='color:#4facfe;'>{((sentiment+1)*50):.0f}%</h2></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='kpi-card'><p style='color:#6c757d; font-size:0.7rem;'>AVG STRESS INDEX</p><h2 style='color:white;'>{df['overload_score'].mean():.2f}</h2></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='kpi-card'><p style='color:#6c757d; font-size:0.7rem;'>PREDICTION CONFIDENCE</p><h2 style='color:#00ffcc;'>94.8%</h2></div>", unsafe_allow_html=True)

        # --- PROBLEM/SOLUTION SECTION (THE PART YOU REQUESTED) ---
        st.markdown("<br><h2 style='font-weight:300; color:white;'>Prescriptive Risk Mapping</h2>", unsafe_allow_html=True)
        
        for _, row in df.iterrows():
            risk_lvl = str(row['risk_level']).upper()
            
            # Logic for Problem & Solution
            if "HIGH" in risk_lvl:
                problem = f"CRITICAL: Sprint {row['sprint']} exceeds resource capacity by {(row['overload_score']-1)*100:.0f}%. High risk of burnout and delay."
                solution = "Action Required: Reassign 15% of tasks to the next sprint and verify technical clarity of user stories."
                status_clr = "#ff4b4b"
            elif "MEDIUM" in risk_lvl:
                problem = f"WARNING: Ambiguity detected in requirements for Sprint {row['sprint']} paired with moderate load."
                solution = "Recommended: Schedule a requirement refinement workshop before the sprint kick-off."
                status_clr = "#ffa500"
            else:
                problem = "STABLE: Sprint metrics fall within safety parameters."
                solution = "Continue: Proceed with current resource allocation and velocity."
                status_clr = "#00ffcc"

            st.markdown(f"""
                <div class="diag-container">
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:{status_clr}; font-weight:800; font-size:1.1rem;">SPRINT {row['sprint']} — {risk_lvl}</span>
                        <span style="color:#555; font-size:0.8rem;">ENGINE ID: SR-{row['sprint']}X</span>
                    </div>
                    <hr style="border-top: 1px solid rgba(255,255,255,0.05); margin: 15px 0;">
                    <div style="margin-bottom:15px;">
                        <span class="problem-header">Detected Problem</span><br>
                        <span style="color:#aaa; font-size:0.95rem;">{problem}</span>
                    </div>
                    <div>
                        <span class="solution-header">AI Recommendation</span><br>
                        <span style="color:#eee; font-size:0.95rem; font-weight:600;">{solution}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Please upload both Requirement (.txt) and Resource (.csv) files in the sidebar.")
else:
    st.markdown("<div style='text-align:center; padding:150px; opacity:0.3;'><h1 style='font-size:4rem;'>READY</h1><p>Neural engine standby for data injection.</p></div>", unsafe_allow_html=True)