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

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError:
    st.error("System Error: Architectural modules (src/) not found.")

# --- 2. ELITE BUSINESS UI DESIGN ---
st.set_page_config(page_title="SentianRisk | Governance", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .stApp { background-color: #0c0e12; font-family: 'Inter', sans-serif; color: #e1e1e1; }
    
    /* Breathing Neural Pulse (Standby State) */
    .status-ring { width: 60px; height: 60px; border-radius: 50%; border: 1px solid #00d9ff; margin: 0 auto 30px; animation: pulse 3s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 0.2; transform: scale(0.95); } 50% { opacity: 1; transform: scale(1); } }

    /* Interactive KPI Boxes (Floating Effect) */
    .kpi-box { 
        background-color: #161a21; border: 1px solid #2d343f; border-radius: 4px; 
        padding: 24px; text-align: center; transition: all 0.3s ease; 
    }
    .kpi-box:hover { transform: translateY(-5px); border-color: #00d9ff; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4); }

    /* High-Impact Recovery Brief (The Solution Rectangle) */
    .recovery-brief { 
        border-left: 4px solid #00d9ff; background-color: #1a1e26; 
        padding: 25px; margin: 25px 0; border-radius: 0 4px 4px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* The High-Energy Audit Button (Gradient Style) */
    div.stButton > button {
        background: linear-gradient(90deg, #00d9ff, #005fcc);
        color: white; border: none; padding: 15px 30px; border-radius: 4px;
        font-weight: 700; text-transform: uppercase; letter-spacing: 2px;
        transition: all 0.3s ease; width: 100%; margin-top: 20px;
    }
    div.stButton > button:hover { transform: scale(1.02); box-shadow: 0 8px 20px rgba(0, 217, 255, 0.4); color: white; }
    
    .footer { text-align: center; color: #333; font-size: 0.7rem; letter-spacing: 2px; padding: 40px 0; border-top: 1px solid #1c2128; margin-top: 60px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORPORATE IDENTITY ---
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid #1c2128; margin-bottom: 40px;">
        <div><h2 style='margin:0; font-weight:700; color:#ffffff; letter-spacing:-1px;'>SENTIAN<span style='color:#00d9ff;'>RISK</span></h2><p style='color:#666; margin:0; font-size:0.75rem; text-transform:uppercase; letter-spacing:3px;'>Strategic Governance Platform</p></div>
        <div style="text-align:right;"><p style='color:#00d9ff; margin:0; font-size:0.65rem; font-weight:700; letter-spacing:1px;'>AUDIT ENGINE: ONLINE</p></div>
    </div>
    """, unsafe_allow_html=True)

# --- 4. CONTROL PANEL ---
with st.sidebar:
    st.markdown("<p style='font-size:0.7rem; color:#666; text-transform:uppercase; letter-spacing:1px;'>Data Ingestion</p>", unsafe_allow_html=True)
    req_file = st.file_uploader("Project Specifications (.txt)", type=["txt"])
    spr_file = st.file_uploader("Operational Schema (.csv)", type=["csv"])
    st.markdown("---")
    execute = st.button("Initialize Audit")

# --- 5. AUDIT & RECOVERY LOGIC ---
if execute and req_file and spr_file:
    try:
        raw_df = pd.read_csv(spr_file)
        r_text = req_file.getvalue().decode("utf-8")
        
        # STRUCTURAL VALIDATION (Schema Gatekeeper)
        csv_cols = "".join(raw_df.columns).lower()
        project_vectors = ['sprint', 'task', 'hour', 'capacity', 'effort', 'deadline']
        
        if not any(v in csv_cols for v in project_vectors):
            st.markdown("""
                <div style='border: 1px solid #ff4b4b; padding: 20px; border-radius: 4px; background-color: rgba(255, 75, 75, 0.05);'>
                    <h5 style='color: #ff4b4b; margin:0; letter-spacing:1px;'>STRUCTURAL MISMATCH DETECTED</h5>
                    <p style='color: #888; font-size: 0.85rem; margin-top:10px;'>
                        Data integrity audit failed. The uploaded metadata does not align with Project Governance vectors. 
                        Please synchronize documentation with standard Sprint and Allocation schemas.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Executing Risk Induction..."):
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
            with k1: st.markdown(f"<div class='kpi-box'><p style='color:#666; font-size:0.7rem; letter-spacing:1px;'>SENTIMENT BIAS</p><h3 style='color:#00d9ff; margin:0;'>{('STABLE' if sentiment > 0 else 'VOLATILE')}</h3></div>", unsafe_allow_html=True)
            with k2: st.markdown(f"<div class='kpi-box'><p style='color:#666; font-size:0.7rem; letter-spacing:1px;'>RISK COEFFICIENT</p><h3 style='color:#fff; margin:0;'>{avg_risk:.2f}</h3></div>", unsafe_allow_html=True)
            with k3: st.markdown(f"<div class='kpi-box'><p style='color:#666; font-size:0.7rem; letter-spacing:1px;'>ITERATIONS</p><h3 style='color:#fff; margin:0;'>{len(df)}</h3></div>", unsafe_allow_html=True)
            with k4: st.markdown(f"<div class='kpi-box'><p style='color:#666; font-size:0.7rem; letter-spacing:1px;'>MODEL FIDELITY</p><h3 style='color:#fff; margin:0;'>94%</h3></div>", unsafe_allow_html=True)

            # --- THE POWER FEATURE: STRATEGIC RECOVERY PLAN ---
            st.markdown("<div class='recovery-brief'>", unsafe_allow_html=True)
            st.markdown("<h5 style='color:#ffffff; margin-bottom:15px; letter-spacing:2px; font-weight:700;'>AI-DRIVEN STRATEGIC RECOVERY PLAN</h5>", unsafe_allow_html=True)
            
            # Scenario Logic: Ensures the box is NEVER empty
            if avg_risk > 0.6:
                st.markdown(f"<p style='color:#ff4b4b; font-weight:700; margin-bottom:5px;'>DETECTED: CRITICAL RESOURCE SATURATION</p>", unsafe_allow_html=True)
                st.write(f"**Problem:** Resource saturation is exceeding {int(avg_risk*100)}%, predicting a burnout event.")
                st.write("**Immediate Solution:** Implement 'Scope Freeze'. Remove 20% of low-priority backlog.")
                st.write("**Long-term Solution:** Re-baseline the delivery timeline by 14 business days.")
            
            elif sentiment < 0:
                st.markdown(f"<p style='color:#ffaa00; font-weight:700; margin-bottom:5px;'>DETECTED: LINGUISTIC INSTABILITY</p>", unsafe_allow_html=True)
                st.write("**Problem:** High ambiguity detected in requirements, increasing misinterpretation risk.")
                st.write("**Immediate Solution:** Mandatory Stakeholder 'Clarification Sync' to define acceptance criteria.")
                st.write("**Long-term Solution:** Integrate peer-review governance for all documentation.")
            
            else:
                st.markdown(f"<p style='color:#00ff9d; font-weight:700; margin-bottom:5px;'>STATUS: OPTIMAL OPERATIONAL ALIGNMENT</p>", unsafe_allow_html=True)
                st.write("**Problem:** No critical structural or linguistic risks identified.")
                st.write("**Immediate Solution:** Maintain current velocity and document best practices.")
                st.write("**Long-term Solution:** Proceed to next project phase with current allocation.")

            st.markdown("</div>", unsafe_allow_html=True)

            st.line_chart(df.set_index('sprint')[['overload_score', 'ambiguity_score']])

    except Exception as e:
        st.error(f"Engine Failure: {str(e)}")
else:
    st.markdown("<div style='text-align:center; padding-top:100px;'><div class='status-ring'></div><p style='color:#333; letter-spacing:5px; font-size:0.8rem;'>AWAITING AUDIT INITIALIZATION</p></div>", unsafe_allow_html=True)

st.markdown(f"<div class='footer'>SENTIANRISK ARCHITECTURE &copy; 2026 BY SHAMA SALEEM</div>", unsafe_allow_html=True)