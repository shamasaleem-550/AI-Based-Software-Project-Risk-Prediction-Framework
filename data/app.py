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

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="SentianRisk AI | Enterprise", layout="wide")

# --- 4. ADVANCED CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(145deg, #16181d, #1a1c23);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 8px 8px 16px #080a0d, -4px -4px 12px #14171d;
    }
    .card-label { color: #4facfe; font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; }
    .template-box {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. HEADER ---
st.markdown("<br>", unsafe_allow_html=True)
t_col1, t_col2 = st.columns([3, 1])
with t_col1:
    st.markdown(" <h1 style='letter-spacing: 3px; font-weight: 300; color: white;'>SENTIAN<span style='font-weight: 800; color: #4facfe;'>RISK</span> AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d;'>Hybrid Decision Support System for Software Risk Orchestration</p>", unsafe_allow_html=True)
with t_col2:
    st.markdown(f"<div style='text-align: right; border-left: 2px solid #4facfe; padding-left: 15px;'><span style='color: #4facfe; font-size: 0.7rem; font-weight: 700;'>RESEARCHER</span><br><span style='color: white;'>SHAMA SALEEM</span></div>", unsafe_allow_html=True)

st.markdown("---")

# --- 6. SIDEBAR & TEMPLATE GENERATOR ---
with st.sidebar:
    st.markdown("<h3 style='font-size: 1.1rem;'>ENGINE CONTROL</h3>", unsafe_allow_html=True)
    req_file = st.file_uploader("Upload Specs (.txt)", type=["txt"])
    spr_file = st.file_uploader("Upload Resources (.csv)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 📥 QUICK TEMPLATES")
    
    if st.button("Generate High-Risk Scenario"):
        st.session_state['sample_req'] = "The system must be built immediately. The requirements are not yet finalized but we must start now."
        st.session_state['sample_csv'] = pd.DataFrame({'sprint':[1,2],'hours_assigned':[80,90],'developer_capacity':[40,40]})
        st.info("High-Risk Template Loaded!")

    if st.button("Generate Stable Scenario"):
        st.session_state['sample_req'] = "The login system shall use OAuth 2.0. The response time must be under 200ms for 1000 concurrent users."
        st.session_state['sample_csv'] = pd.DataFrame({'sprint':[1,2],'hours_assigned':[30,35],'developer_capacity':[40,40]})
        st.success("Stable Template Loaded!")

    st.markdown("---")
    if st.button("🚀 EXECUTE ANALYSIS", use_container_width=True):
        st.session_state['active'] = True

# --- 7. MAIN DASHBOARD ---
if st.session_state.get('active'):
    # Logic to use uploaded file OR sample data
    if req_file: r_text = req_file.getvalue().decode("utf-8")
    else: r_text = st.session_state.get('sample_req', "")

    if spr_file: r_df = pd.read_csv(spr_file)
    else: r_df = st.session_state.get('sample_csv', None)

    if not r_text or r_df is None:
        st.warning("Please upload files or select a Template in the sidebar.")
    else:
        # Save and Run Backend
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
        r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        with st.spinner("Synchronizing Linguistic and Numerical Risk Vectors..."):
            create_combined_dataset()
            train_hybrid_model()

        res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            
            # --- DASHBOARD RESULTS ---
            st.markdown("### 📊 RISK METRICS")
            m_cols = st.columns(len(df))
            for i, (_, row) in enumerate(df.iterrows()):
                with m_cols[i]:
                    lvl = str(row['risk_level']).upper()
                    clr = "#FF4B4B" if "HIGH" in lvl else "#FFA500" if "MEDIUM" in lvl else "#00FF00"
                    st.markdown(f"<div class='metric-card'><div class='card-label'>SPRINT {row['sprint']}</div><h2 style='color:{clr};'>{lvl}</h2></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("### 📈 ANALYTIC TREND")
                st.line_chart(df[['ambiguity_score', 'overload_score']])
            with c2:
                st.markdown("### 🧠 AI REASONER")
                for _, row in df.iterrows():
                    with st.expander(f"Sprint {row['sprint']} Deep Dive"):
                        st.write(f"**NLP Ambiguity:** {row['ambiguity_score']:.2f}")
                        st.write(f"**Resource Strain:** {row['overload_score']:.2f}")
                        st.progress(min(row['overload_score'], 1.2)/1.2)
else:
    st.markdown("""
        <div style='text-align: center; padding: 100px;'>
            <h2 style='font-weight: 300; color: #6c757d;'>Engine Standby</h2>
            <p style='color: #444;'>Select a 'Quick Template' or upload custom data to begin.</p>
        </div>
    """, unsafe_allow_html=True)