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
    st.error(f"Module import failed: {e}")

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="SentianRisk AI", layout="wide")

# --- 4. THE "PREMIUM DARK" CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0E1117; }

    /* The High-End Neumorphic Card */
    .metric-card {
        background: linear-gradient(145deg, #1e2129, #16181d);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 40px 20px;
        text-align: center;
        box-shadow: 12px 12px 24px #080a0d, -8px -8px 24px #1a1d24;
        transition: all 0.4s ease;
    }
    
    .metric-card:hover { transform: translateY(-5px); border: 1px solid #4facfe; }

    /* AI Prescription Box */
    .ai-box {
        background: rgba(79, 172, 254, 0.05);
        border-left: 5px solid #4facfe;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    .problem-text { color: #FF4B4B; font-weight: 700; font-size: 0.9rem; }
    .solution-text { color: #00FF00; font-weight: 700; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 5. EXECUTIVE HEADER ---
st.markdown("<br>", unsafe_allow_html=True)
h_col1, h_col2 = st.columns([4, 1])

with h_col1:
    st.markdown("<h1 style='font-weight:300; margin:0;'>SENTIAN<span style='font-weight:800; color:#4facfe;'>RISK</span> AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d; font-size:1.1rem;'>Advanced Predictive Governance & Risk Orchestration</p>", unsafe_allow_html=True)

with h_col2:
    st.markdown(f"<div style='text-align:right; border-right:3px solid #4facfe; padding-right:20px;'><span style='color:#4facfe; font-size:0.7rem; font-weight:800;'>DESIGNER</span><br><span style='color:white; font-size:1.1rem; font-weight:300;'>SHAMA SALEEM</span></div>", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.05); margin-bottom: 40px;'>", unsafe_allow_html=True)

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("<h2 style='font-size:1.2rem; color:#4facfe;'>CONTROL CENTER</h2>", unsafe_allow_html=True)
    req_file = st.file_uploader("Requirement Specs", type=["txt"])
    spr_file = st.file_uploader("Sprint Data", type=["csv"])
    
    st.markdown("---")
    if st.button("RUN AI ENGINE", type="primary", use_container_width=True):
        st.session_state['active'] = True

# --- 7. MAIN DASHBOARD ---
if st.session_state.get('active'):
    if req_file and spr_file:
        # Save & Process
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "wb") as f: f.write(req_file.getvalue())
        pd.read_csv(spr_file).to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        with st.spinner("AI Brain Initializing..."):
            create_combined_dataset()
            train_hybrid_model()

        df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))

        # --- AI FEATURE 1: PREDICTIVE METRIC CARDS ---
        st.markdown("<h3 style='font-weight:300; color:#6c757d;'>Executive Risk Outlook</h3>", unsafe_allow_html=True)
        m_cols = st.columns(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            with m_cols[i]:
                lvl = str(row['risk_level']).upper()
                clr = "#FF4B4B" if "HIGH" in lvl else "#FFA500" if "MEDIUM" in lvl else "#00FF00"
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="color:#888; font-size:0.7rem; letter-spacing:2px; margin-bottom:10px;">SPRINT {row['sprint']}</div>
                        <h2 style="color:{clr}; font-weight:800; margin:0;">{lvl}</h2>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- AI FEATURE 2: THE PROBLEM/SOLUTION PRESCRIPTION ---
        st.markdown("### 🧠 AI Prescription Engine")
        st.markdown("<p style='color:#6c757d;'>Autonomous identification of project bottlenecks and mitigation strategies.</p>", unsafe_allow_html=True)
        
        for _, row in df.iterrows():
            risk_val = str(row['risk_level']).upper()
            prob = "Resource Overload" if row['overload_score'] > 1.1 else "Linguistic Ambiguity" if row['ambiguity_score'] > 0.5 else "None"
            sol = "Redistribute tasks to Sprint +1" if "Resource" in prob else "Re-draft SRS documentation" if "Linguistic" in prob else "Continue at current velocity"
            
            st.markdown(f"""
                <div class="ai-box">
                    <span style="font-size:1.1rem; font-weight:600; color:white;">Insight: Sprint {row['sprint']} Analysis</span><br>
                    <span class="problem-text">PROBLEM:</span> <span style="color:#ddd;">{prob} detected.</span> | 
                    <span class="solution-text">SOLUTION:</span> <span style="color:#4facfe;">{sol}</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- AI FEATURE 3: DUAL-VECTOR ANALYTICS ---
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 📈 Risk Trajectory Graph")
            st.line_chart(df[['ambiguity_score', 'overload_score']])
        with c2:
            st.markdown("### 🛰️ Factor Density")
            st.bar_chart(df[['overload_score']])
            st.caption("Resource stress measured against developer capacity thresholds.")

else:
    st.markdown("<div style='text-align:center; padding:150px; border:1px dashed #222; border-radius:40px;'><h2 style='font-weight:200; color:#444;'>AI ENGINE STANDBY</h2><p style='color:#333;'>Awaiting input data stream...</p></div>", unsafe_allow_html=True)