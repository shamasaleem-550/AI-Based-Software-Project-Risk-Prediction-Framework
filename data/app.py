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

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="SentianRisk AI", layout="wide")

# --- 4. AESTHETIC CSS (Glassmorphism & Neumorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0E1117;
    }

    /* Professional Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #16181d, #1a1c23);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        box-shadow: 8px 8px 16px #080a0d, -4px -4px 12px #14171d;
        transition: all 0.4s ease;
    }
    
    .metric-card:hover {
        border: 1px solid rgba(79, 172, 254, 0.4);
        transform: translateY(-8px);
    }

    /* Subtext for cards */
    .card-label {
        color: #6c757d;
        font-size: 0.8rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: #0B0D11 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. CLEAN HEADER (NO STICKERS) ---
st.markdown("<br>", unsafe_allow_html=True)
col_t1, col_t2 = st.columns([3, 1])

with col_t1:
    st.markdown("""
        <h1 style='letter-spacing: 3px; font-weight: 300; margin-bottom: 0px; color: white;'>
            SENTIAN<span style='font-weight: 800; color: #4facfe;'>RISK</span> AI
        </h1>
        <p style='color: #6c757d; font-size: 1rem; margin-top: 0px; font-weight: 400;'>
            HYBRID PREDICTIVE INTELLIGENCE FOR AGILE FRAMEWORKS
        </p>
    """, unsafe_allow_html=True)

with col_t2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='text-align: right; border-left: 2px solid #4facfe; padding-left: 15px;'>
            <span style='color: #4facfe; font-size: 0.8rem; font-weight: 700;'>RESEARCHER</span><br>
            <span style='color: white; font-size: 1rem; font-weight: 300;'>SHAMA SALEEM</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.05); margin-top: 30px; margin-bottom: 40px;'>", unsafe_allow_html=True)

# --- 6. SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("<h2 style='font-weight: 800; font-size: 1.2rem;'>CONTROL PANEL</h2>", unsafe_allow_html=True)
    st.write("Upload project metadata for heuristic processing.")
    
    req_file = st.file_uploader("Project Requirements (.txt)", type=["txt"])
    spr_file = st.file_uploader("Resource Data (.csv)", type=["csv"])
    
    st.markdown("---")
    if st.button("RUN ENGINE", use_container_width=True):
        if req_file and spr_file:
            st.session_state['ready'] = True
        else:
            st.error("Please provide both datasets.")

# --- 7. DASHBOARD LOGIC ---
if st.session_state.get('ready'):
    try:
        # File System Operations
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "wb") as f:
            f.write(req_file.getvalue())
        with open(os.path.join(root_path, "data", "sprint_tasks.csv"), "wb") as f:
            f.write(spr_file.getvalue())

        with st.spinner("Decoding Linguistic and Resource Patterns..."):
            create_combined_dataset()
            train_hybrid_model()

        res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            
            # Metric Card Display
            st.markdown("<h3 style='font-weight: 400; color: #6c757d;'>EXECUTIVE SUMMARY</h3>", unsafe_allow_html=True)
            cols = st.columns(len(df))
            
            for i, (_, row) in enumerate(df.iterrows()):
                with cols[i]:
                    risk = str(row['risk_level']).strip().upper()
                    if "HIGH" in risk: color = "#FF4B4B"
                    elif "MEDIUM" in risk: color = "#FFA500"
                    else: color = "#00FF00"
                    
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="card-label">Sprint {row['sprint']}</div>
                            <h2 style="color: {color}; font-weight: 800; letter-spacing: 1px; margin: 0;">{risk}</h2>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Analytics Section
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("<h3 style='font-weight: 400; color: #6c757d;'>RISK TRAJECTORY</h3>", unsafe_allow_html=True)
                st.line_chart(df[['ambiguity_score', 'overload_score']])
            with c2:
                st.markdown("<h3 style='font-weight: 400; color: #6c757d;'>AI REASONING</h3>", unsafe_allow_html=True)
                for _, row in df.iterrows():
                    with st.expander(f"Sprint {row['sprint']} Insights"):
                        st.write(f"**Linguistic Ambiguity:** {row['ambiguity_score']:.2f}")
                        st.write(f"**Resource Load:** {row['overload_score']:.2f}")
                        st.progress(min(row['overload_score'], 1.0))
        
    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.markdown("""
        <div style='text-align: center; padding: 100px; color: #444;'>
            <h2 style='font-weight: 300;'>Engine Standby</h2>
            <p>Awaiting data upload from the Control Panel to initialize analysis.</p>
        </div>
    """, unsafe_allow_html=True)