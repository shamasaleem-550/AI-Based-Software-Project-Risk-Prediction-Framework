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

# --- 3. ADVANCED DATA VALIDATION ---
def validate_inputs(req_text, sprint_df):
    errors = []
    if len(req_text.strip()) < 15:
        errors.append("Requirement text is too brief for NLP ambiguity scoring.")
    
    required_columns = ['sprint', 'hours_assigned', 'developer_capacity']
    for col in required_columns:
        if col not in sprint_df.columns:
            errors.append(f"Missing mandatory column: '{col}'")
    return errors

# --- 4. PAGE CONFIGURATION ---
st.set_page_config(page_title="SentianRisk AI | Advanced", layout="wide")

# --- 5. PREMIUM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0E1117; }

    /* Advanced Neumorphic Cards */
    .metric-card {
        background: linear-gradient(145deg, #16181d, #1a1c23);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 8px 8px 16px #080a0d, -4px -4px 12px #14171d;
    }
    .card-label {
        color: #4facfe;
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    /* Status Badge */
    .status-badge {
        padding: 5px 12px;
        border-radius: 50px;
        font-size: 0.7rem;
        font-weight: 700;
        background: rgba(79, 172, 254, 0.1);
        color: #4facfe;
        border: 1px solid #4facfe;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 6. CLEAN HEADER ---
st.markdown("<br>", unsafe_allow_html=True)
col_t1, col_t2 = st.columns([3, 1])

with col_t1:
    st.markdown("""
        <h1 style='letter-spacing: 3px; font-weight: 300; margin-bottom: 0px; color: white;'>
            SENTIAN<span style='font-weight: 800; color: #4facfe;'>RISK</span> AI
        </h1>
        <p style='color: #6c757d; font-size: 0.9rem; margin-top: 0px;'>
            <span class="status-badge">ADVANCED HYBRID MODEL v2.0</span> 
            &nbsp; Predictive Governance for Agile Teams
        </p>
    """, unsafe_allow_html=True)

with col_t2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='text-align: right; border-left: 2px solid #4facfe; padding-left: 15px;'>
            <span style='color: #4facfe; font-size: 0.7rem; font-weight: 700;'>LEAD RESEARCHER</span><br>
            <span style='color: white; font-size: 0.9rem; font-weight: 300;'>SHAMA SALEEM</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.05); margin-bottom: 35px;'>", unsafe_allow_html=True)

# --- 7. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='font-weight: 800; font-size: 1.1rem;'>ENGINE CONTROL</h2>", unsafe_allow_html=True)
    req_file = st.file_uploader("Requirements Specification (.txt)", type=["txt"])
    spr_file = st.file_uploader("Resource Allocation (.csv)", type=["csv"])
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.75rem; color: #6c757d; font-weight: 700;'>VALIDATION TEMPLATES</p>", unsafe_allow_html=True)
    
    # Requirements Template
    s_txt = "System must handle concurrent users. The interface needs to be fast. Security is a priority but not fully defined."
    st.download_button("📥 Req Template", s_txt, "req_template.txt", use_container_width=True)

    # CSV Template
    s_df = pd.DataFrame({
        'sprint': [1, 2, 3],
        'task_name': ['Frontend', 'Backend', 'DevOps'],
        'hours_assigned': [35, 50, 45],
        'developer_capacity': [40, 40, 40]
    })
    st.download_button("📥 CSV Template", s_df.to_csv(index=False), "res_template.csv", use_container_width=True)
    
    st.markdown("---")
    if st.button("EXECUTE ADVANCED ANALYSIS", use_container_width=True):
        if req_file and spr_file:
            st.session_state['active'] = True
        else:
            st.error("Input missing.")

# --- 8. DASHBOARD MAIN VIEW ---
if st.session_state.get('active'):
    try:
        r_text = req_file.getvalue().decode("utf-8")
        r_df = pd.read_csv(spr_file)
        
        # Validation Layer
        errs = validate_inputs(r_text, r_df)
        if errs:
            for e in errs: st.error(f"⚠️ {e}")
        else:
            # Data Persistence
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
            r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

            with st.spinner("Calculating Weighted Risk Coefficients..."):
                create_combined_dataset()
                train_hybrid_model() # This now runs your 70/30 weighted logic

            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            if os.path.exists(res_path):
                df = pd.read_csv(res_path)
                
                # --- Metrics Row ---
                st.markdown("<h3 style='font-weight: 400; color: #6c757d;'>RISK ASSESSMENT SUMMARY</h3>", unsafe_allow_html=True)
                cols = st.columns(len(df))
                
                for i, (_, row) in enumerate(df.iterrows()):
                    with cols[i]:
                        risk = str(row['risk_level']).strip().upper()
                        # Color logic for High/Medium/Low
                        if "HIGH" in risk: color = "#FF4B4B"
                        elif "MEDIUM" in risk: color = "#FFA500"
                        else: color = "#00FF00"
                        
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="card-label">SPRINT {row['sprint']}</div>
                                <h2 style="color: {color}; font-weight: 800; margin: 0; font-size: 1.5rem;">{risk}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # --- Analytics Row ---
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown("<h4 style='font-weight: 400; color: #6c757d;'>HYBRID RISK TRAJECTORY</h4>", unsafe_allow_html=True)
                    st.line_chart(df[['ambiguity_score', 'overload_score']])
                
                with c2:
                    st.markdown("<h4 style='font-weight: 400; color: #6c757d;'>AI LOGIC DECODER</h4>", unsafe_allow_html=True)
                    st.caption("Weighting: 70% Resource Load | 30% NLP Ambiguity")
                    for _, row in df.iterrows():
                        with st.expander(f"Sprint {row['sprint']} Diagnostics"):
                            st.write(f"**Linguistic Risk:** {row['ambiguity_score']:.2f}")
                            st.write(f"**Resource Strain:** {row['overload_score']:.2f}")
                            # Visualizing the strain
                            st.progress(min(row['overload_score'], 1.0))
                            if row['overload_score'] > 1.1:
                                st.warning("Critical Capacity Breach Detected.")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.markdown("""
        <div style='text-align: center; padding: 120px; color: #444; border: 1px dashed rgba(255,255,255,0.05); border-radius: 30px;'>
            <h2 style='font-weight: 300;'>Engine Standby</h2>
            <p>Awaiting validated data stream to initialize predictive models.</p>
        </div>
    """, unsafe_allow_html=True)