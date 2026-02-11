import sys
import os
import pandas as pd
import streamlit as st
from textblob import TextBlob
import nltk

# --- 1. SYSTEM INITIALIZATION ---
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
    st.error("Engine components not found in /src.")

# --- 2. CONFIG & THEME ---
st.set_page_config(page_title="SentianRisk | Decision Support", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp { background-color: #0c0e11; font-family: 'Inter', sans-serif; }
    
    /* Executive Header */
    .header-box {
        padding: 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 30px;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }

    /* Problem/Solution Blocks */
    .diagnostic-card {
        background: #161a1f;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .status-tag {
        font-size: 0.7rem;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .problem-label { color: #ff5f5f; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; }
    .solution-label { color: #00d9ff; font-weight: 700; font-size: 0.8rem; text-transform: uppercase; }
    
    /* Footer Signature */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0c0e11;
        color: #444;
        text-align: center;
        padding: 10px;
        font-size: 0.7rem;
        border-top: 1px solid rgba(255,255,255,0.03);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. UI HEADER ---
st.markdown("""
    <div class="header-box">
        <h1 style='color:white; margin:0; font-weight:400;'>SENTIAN<span style='color:#00d9ff; font-weight:700;'>RISK</span></h1>
        <span style='color:#555; font-size:0.8rem; font-weight:600;'>INTELLIGENT RISK ORCHESTRATION</span>
    </div>
    """, unsafe_allow_html=True)

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h4 style='color:white; margin-bottom:20px;'>Data Ingestion</h4>", unsafe_allow_html=True)
    req_file = st.file_uploader("Requirement Specs (.txt)", type=["txt"])
    spr_file = st.file_uploader("Sprint Metadata (.csv)", type=["csv"])
    
    st.markdown("---")
    if st.button("PROCESS MODEL", type="primary", use_container_width=True):
        st.session_state['active'] = True

# --- 5. DASHBOARD MAIN ---
if st.session_state.get('active'):
    if req_file and spr_file:
        r_text = req_file.getvalue().decode("utf-8")
        r_df = pd.read_csv(spr_file)
        
        # Save & Run
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
        r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        create_combined_dataset()
        train_hybrid_model()
        df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))

        st.markdown("<h3 style='font-weight:400; color:#888; margin-bottom:25px;'>Strategic Diagnostics</h3>", unsafe_allow_html=True)

        # --- PROBLEM/SOLUTION DISPLAY ---
        for _, row in df.iterrows():
            risk = str(row['risk_level']).upper()
            
            # Map Logic
            if "HIGH" in risk:
                tag_bg, tag_txt = "rgba(255, 95, 95, 0.1)", "#ff5f5f"
                prob = f"Significant resource saturation detected in Sprint {row['sprint']} (Stress: {row['overload_score']:.2f})."
                sol = "Immediate load redistribution required. Consider deferring low-priority user stories to the next cycle."
            elif "MEDIUM" in risk:
                tag_bg, tag_txt = "rgba(255, 165, 0, 0.1)", "#ffa500"
                prob = "Linguistic ambiguity detected. Requirements may lead to implementation variance."
                sol = "Schedule a technical alignment meeting to define granular acceptance criteria."
            else:
                tag_bg, tag_txt = "rgba(0, 217, 255, 0.1)", "#00d9ff"
                prob = "Nominal performance metrics. Risk factors are within acceptable deviation."
                sol = "Maintain current sprint velocity. Proceed with standard monitoring."

            st.markdown(f"""
                <div class="diagnostic-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                        <b style="color:white; font-size:1.1rem;">Sprint {row['sprint']} Analysis</b>
                        <span class="status-tag" style="background:{tag_bg}; color:{tag_txt}; border:1px solid {tag_txt};">{risk}</span>
                    </div>
                    <div style="margin-bottom:12px;">
                        <span class="problem-label">Detected Problem</span><br>
                        <span style="color:#aaa; font-size:0.95rem;">{prob}</span>
                    </div>
                    <div>
                        <span class="solution-label">AI Prescription</span><br>
                        <span style="color:white; font-size:0.95rem;">{sol}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Awaiting file upload...")
else:
    # Minimalist Idle State
    st.markdown("""
        <div style='text-align:center; padding-top:100px;'>
            <h2 style='color:#222; font-weight:700;'>SYSTEM STANDBY</h2>
            <p style='color:#333;'>Neural engine initialized and awaiting data stream.</p>
        </div>
    """, unsafe_allow_html=True)

# --- 6. FOOTER SIGNATURE ---
st.markdown("""
    <div class="footer">
        PROJECT DESIGN & ENGINEERING BY SHAMA SALEEM &nbsp; | &nbsp; © 2026 SENTIANRISK AI
    </div>
    """, unsafe_allow_html=True)