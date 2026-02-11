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
st.set_page_config(page_title="SentianRisk AI", layout="wide", page_icon="🛡️")

# --- 4. CUSTOM CSS (The "Classic Professional" Look) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Custom Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        border: 1px solid rgba(255, 255, 255, 0.3);
        transform: translateY(-5px);
    }
    
    /* Titles and Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Clean Divider */
    hr {
        margin: 2em 0;
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. TOP NAVIGATION/HEADER ---
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.title("🛡️ SentianRisk AI")
    st.caption("Intelligent Framework for Software Project Risk Prediction")
with col_t2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Researcher:** `SHAMA SALEEM`", unsafe_allow_html=True)

st.markdown("---")

# --- 6. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.header("Control Panel")
    st.write("Upload your technical documentation to begin analysis.")
    
    req_file = st.file_uploader("Requirement Specification (.txt)", type=["txt"])
    spr_file = st.file_uploader("Sprint Workload (.csv)", type=["csv"])
    
    if st.button("🚀 Execute AI Analysis", use_container_width=True):
        if req_file and spr_file:
            st.session_state['run_analysis'] = True
        else:
            st.error("Missing files.")

# --- 7. MAIN DASHBOARD ---
if st.session_state.get('run_analysis'):
    try:
        # Save and Run Backend
        with open(os.path.join(root_path, "data", "requirements.txt"), "wb") as f:
            f.write(req_file.getvalue())
        with open(os.path.join(root_path, "data", "sprint_tasks.csv"), "wb") as f:
            f.write(spr_file.getvalue())

        with st.spinner("Processing Hybrid Risk Metrics..."):
            create_combined_dataset()
            train_hybrid_model()

        res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            
            # Metric Row
            st.subheader("Executive Summary")
            cols = st.columns(len(df))
            
            for i, (_, row) in enumerate(df.iterrows()):
                with cols[i]:
                    risk = str(row['risk_level']).strip().upper()
                    # Assigning Icon and color based on logic
                    if "HIGH" in risk: color, icon = "#FF4B4B", "🔴"
                    elif "MEDIUM" in risk: color, icon = "#FFA500", "🟠"
                    else: color, icon = "#00FF00", "🟢"
                    
                    # Classic HTML Card
                    st.markdown(f"""
                        <div class="metric-card">
                            <p style="color: #888; margin-bottom: 5px; font-size: 0.9em;">SPRINT {row['sprint']}</p>
                            <h2 style="color: {color}; margin: 0;">{icon} {risk}</h2>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Analytics Row
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("Risk Distribution Trend")
                st.line_chart(df[['ambiguity_score', 'overload_score']])
            with c2:
                st.subheader("AI Decision Logic")
                for _, row in df.iterrows():
                    with st.expander(f"Analysis: Sprint {row['sprint']}"):
                        st.write(f"**Linguistic Risk:** {row['ambiguity_score']:.2f}")
                        st.write(f"**Capacity Risk:** {row['overload_score']:.2f}")
                        st.progress(row['overload_score'])
        
    except Exception as e:
        st.error(f"Analysis Interrupted: {e}")
else:
    st.info("👋 Welcome. Please upload your project data in the sidebar to generate the risk dashboard.")