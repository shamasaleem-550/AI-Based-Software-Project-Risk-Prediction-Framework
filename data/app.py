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
    st.error(f"Module import failed. Error: {e}")

# --- 3. DATA VALIDATION LOGIC ---
def validate_inputs(req_text, sprint_df):
    """Checks if the data is healthy for the AI to process."""
    errors = []
    if len(req_text.strip()) < 10:
        errors.append("Requirement text is too short for linguistic analysis.")
    
    required_columns = ['sprint', 'hours_assigned', 'developer_capacity']
    for col in required_columns:
        if col not in sprint_df.columns:
            errors.append(f"Missing required column: '{col}'")
    return errors

# --- 4. PAGE CONFIGURATION ---
st.set_page_config(page_title="SentianRisk AI", layout="wide")

# --- 5. AESTHETIC CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0E1117; }

    /* Premium Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #16181d, #1a1c23);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 8px 8px 16px #080a0d, -4px -4px 12px #14171d;
        transition: all 0.4s ease;
    }
    .metric-card:hover {
        border: 1px solid rgba(79, 172, 254, 0.4);
        transform: translateY(-5px);
    }
    .card-label {
        color: #6c757d;
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0B0D11 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. HEADER ---
st.markdown("<br>", unsafe_allow_html=True)
col_t1, col_t2 = st.columns([3, 1])

with col_t1:
    st.markdown("""
        <h1 style='letter-spacing: 3px; font-weight: 300; margin-bottom: 0px; color: white;'>
            SENTIAN<span style='font-weight: 800; color: #4facfe;'>RISK</span> AI
        </h1>
        <p style='color: #6c757d; font-size: 0.9rem; margin-top: 0px; font-weight: 400;'>
            HYBRID PREDICTIVE INTELLIGENCE FOR PROJECT GOVERNANCE
        </p>
    """, unsafe_allow_html=True)

with col_t2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='text-align: right; border-left: 2px solid #4facfe; padding-left: 15px;'>
            <span style='color: #4facfe; font-size: 0.7rem; font-weight: 700;'>RESEARCHER</span><br>
            <span style='color: white; font-size: 0.9rem; font-weight: 300;'>SHAMA SALEEM</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.05); margin-bottom: 30px;'>", unsafe_allow_html=True)

# --- 7. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<h2 style='font-weight: 800; font-size: 1.1rem;'>CONTROL PANEL</h2>", unsafe_allow_html=True)
    req_file = st.file_uploader("Upload Requirements (.txt)", type=["txt"])
    spr_file = st.file_uploader("Upload Resource CSV (.csv)", type=["csv"])
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; color: #6c757d; font-weight: 700;'>DEMO TEMPLATES</p>", unsafe_allow_html=True)
    
    # Template: Requirements
    sample_txt = "The system must allow secure user login. Interface should be responsive and robust. Resource usage needs optimization."
    st.download_button("📥 Req Template", sample_txt, "requirements_template.txt", "text/plain", use_container_width=True)

    # Template: CSV
    sample_df = pd.DataFrame({
        'sprint': [1, 2, 3],
        'task_name': ['Auth Module', 'Data Schema', 'API Cloud'],
        'hours_assigned': [38, 45, 52],
        'developer_capacity': [40, 40, 40]
    })
    st.download_button("📥 CSV Template", sample_df.to_csv(index=False), "resource_template.csv", "text/csv", use_container_width=True)
    
    st.markdown("---")
    if st.button("RUN AI ENGINE", use_container_width=True):
        if req_file and spr_file:
            st.session_state['run_ai'] = True
        else:
            st.error("Upload both files to start.")

# --- 8. DASHBOARD MAIN VIEW ---
if st.session_state.get('run_ai'):
    try:
        # Load and Decode
        raw_text = req_file.getvalue().decode("utf-8")
        raw_df = pd.read_csv(spr_file)
        
        # VALIDATION
        v_errors = validate_inputs(raw_text, raw_df)
        if v_errors:
            for err in v_errors: st.error(f"🚨 {err}")
        else:
            # Data Preview Section
            with st.expander("🔍 Input Data Preview"):
                p1, p2 = st.columns(2)
                p1.text_area("Requirements Text", raw_text[:200] + "...", height=150)
                p2.dataframe(raw_df.head(), use_container_width=True)

            # Execution
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f:
                f.write(raw_text)
            raw_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

            with st.spinner("Processing NLP & Heuristic Metrics..."):
                create_combined_dataset()
                train_hybrid_model()

            # Results Display
            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            if os.path.exists(res_path):
                df = pd.read_csv(res_path)
                
                st.markdown("<h3 style='font-weight: 400; color: #6c757d;'>EXECUTIVE SUMMARY</h3>", unsafe_allow_html=True)
                cols = st.columns(len(df))
                
                for i, (_, row) in enumerate(df.iterrows()):
                    with cols[i]:
                        risk = str(row['risk_level']).strip().upper()
                        color = "#FF4B4B" if "HIGH" in risk else "#FFA500" if "MEDIUM" in risk else "#00FF00"
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-label">Sprint {row['sprint']}</div>
                                <h2 style="color: {color}; font-weight: 800; margin: 0;">{risk}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("<h4 style='font-weight: 400; color: #6c757d;'>RISK TRAJECTORY</h4>", unsafe_allow_html=True)
                    st.line_chart(df[['ambiguity_score', 'overload_score']])
                with c2:
                    st.markdown("<h4 style='font-weight: 400; color: #6c757d;'>AI REASONER</h4>", unsafe_allow_html=True)
                    for _, row in df.iterrows():
                        with st.expander(f"Sprint {row['sprint']} Deep Dive"):
                            st.write(f"**NLP Ambiguity:** {row['ambiguity_score']:.2f}")
                            st.write(f"**Resource Load:** {row['overload_score']:.2f}")
                            st.progress(min(row['overload_score'], 1.0))
    except Exception as e:
        st.error(f"System Crash: {e}")
else:
    st.markdown("""
        <div style='text-align: center; padding: 100px; color: #444; border: 1px dashed rgba(255,255,255,0.1); border-radius: 20px;'>
            <h2 style='font-weight: 300;'>Engine Standby</h2>
            <p>Upload files in the Control Panel to initialize validation and analysis.</p>
        </div>
    """, unsafe_allow_html=True)