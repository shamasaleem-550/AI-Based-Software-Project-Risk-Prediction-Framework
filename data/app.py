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

# --- 4. THE AESTHETIC CSS (The "Beauty" Layer) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
    }

    /* Professional Glass-Card Design */
    .metric-card {
        background: linear-gradient(145deg, #1e2129, #16181d);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 40px 20px;
        text-align: center;
        box-shadow: 12px 12px 24px #080a0d, -8px -8px 24px #1a1d24;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .metric-card:hover {
        transform: translateY(-10px);
        border: 1px solid rgba(79, 172, 254, 0.5);
        box-shadow: 0px 15px 30px rgba(0,0,0,0.5);
    }

    .card-label {
        color: #888;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
        font-weight: 600;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: #0B0D11 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.03);
    }

    /* Clean titles */
    h1, h2, h3 {
        color: white !important;
        letter-spacing: -1px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. MINIMALIST HEADER ---
st.markdown("<br>", unsafe_allow_html=True)
h_col1, h_col2 = st.columns([4, 1])

with h_col1:
    st.markdown("""
        <h1 style='font-weight: 300; margin-bottom: 0px;'>
            SENTIAN<span style='font-weight: 800; color: #4facfe;'>RISK</span> AI
        </h1>
        <p style='color: #6c757d; font-size: 1rem; letter-spacing: 0.5px;'>
            Hybrid Predictive Intelligence for Engineering Governance
        </p>
    """, unsafe_allow_html=True)

with h_col2:
    st.markdown(f"""
        <div style='text-align: right; border-right: 3px solid #4facfe; padding-right: 20px;'>
            <span style='color: #4facfe; font-size: 0.7rem; font-weight: 800;'>LEAD</span><br>
            <span style='color: white; font-size: 1.1rem; font-weight: 300;'>SHAMA SALEEM</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.05); margin-bottom: 50px;'>", unsafe_allow_html=True)

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("<h2 style='font-size: 1.2rem; font-weight: 700; color: #4facfe;'>CONTROL CENTER</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6c757d; font-size: 0.85rem;'>Configure your predictive engine parameters below.</p>", unsafe_allow_html=True)
    
    req_file = st.file_uploader("Requirement Specification", type=["txt"])
    spr_file = st.file_uploader("Sprint Metadata", type=["csv"])
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.7rem; color: #555;'>QUICK ACTIONS</p>", unsafe_allow_html=True)
    
    if st.button("Load Demo Scenario", use_container_width=True):
        st.session_state['demo_txt'] = "The system shall process payments securely. Response must be under 2s."
        st.session_state['demo_csv'] = pd.DataFrame({'sprint':[1,2],'hours_assigned':[35,45],'developer_capacity':[40,40]})
        st.toast("Scenario Loaded")

    if st.button("RUN ENGINE", type="primary", use_container_width=True):
        st.session_state['run_now'] = True

# --- 7. DASHBOARD CONTENT ---
if st.session_state.get('run_now'):
    # Determine Data Source
    txt_data = req_file.getvalue().decode("utf-8") if req_file else st.session_state.get('demo_txt', "")
    csv_data = pd.read_csv(spr_file) if spr_file else st.session_state.get('demo_csv', None)

    if not txt_data or csv_data is None:
        st.error("Engine requires data input to initialize.")
    else:
        # Processing
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(txt_data)
        csv_data.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        with st.spinner("Decoding Risk Patterns..."):
            create_combined_dataset()
            train_hybrid_model()

        res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            
            # --- Result Cards ---
            st.markdown("<h3 style='font-weight: 300; color: #6c757d; margin-bottom: 30px;'>Executive Risk Summary</h3>", unsafe_allow_html=True)
            cols = st.columns(len(df))
            for i, (_, row) in enumerate(df.iterrows()):
                with cols[i]:
                    lvl = str(row['risk_level']).upper()
                    # High contrast colors
                    clr = "#FF4B4B" if "HIGH" in lvl else "#FFA500" if "MEDIUM" in lvl else "#00FF00"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="card-label">Sprint {row['sprint']} Status</div>
                            <h2 style="color: {clr}; font-weight: 800; font-size: 1.8rem; margin: 0;">{lvl}</h2>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # --- Analytic Charts ---
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("<h4 style='font-weight: 300; color: #6c757d;'>Risk Factor Distribution</h4>", unsafe_allow_html=True)
                st.line_chart(df[['ambiguity_score', 'overload_score']])
            with c2:
                st.markdown("<h4 style='font-weight: 300; color: #6c757d;'>AI Decision Logic</h4>", unsafe_allow_html=True)
                for _, row in df.iterrows():
                    with st.expander(f"Sprint {row['sprint']} Diagnostics"):
                        st.write(f"**Linguistic Ambiguity:** {row['ambiguity_score']:.2f}")
                        st.write(f"**Resource Capacity:** {row['overload_score']:.2f}")
                        st.progress(min(row['overload_score'], 1.0))
else:
    st.markdown("""
        <div style='text-align: center; padding: 150px; border: 1px dashed rgba(255,255,255,0.05); border-radius: 40px;'>
            <h2 style='font-weight: 200; color: #444;'>ENGINE STANDBY</h2>
            <p style='color: #333;'>Please upload specifications to visualize the risk landscape.</p>
        </div>
    """, unsafe_allow_html=True)