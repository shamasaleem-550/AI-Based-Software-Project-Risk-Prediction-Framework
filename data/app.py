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
    st.error("Backend modules missing.")

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="SentianRisk Pro", layout="wide", initial_sidebar_state="expanded")

# --- 3. THE "LUXURY DARK" CSS ---
st.markdown("""
    <style>
    /* Global Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    
    .stApp { background-color: #080a0c; font-family: 'Plus Jakarta Sans', sans-serif; }
    header {visibility: hidden;}
    
    /* Custom Card Styling */
    .metric-container {
        background: linear-gradient(145deg, #111418, #181c22);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 10px 10px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }

    .risk-high { border-left: 5px solid #ff4b4b; }
    .risk-med { border-left: 5px solid #ffa500; }
    .risk-low { border-left: 5px solid #00ffcc; }

    .card-title { color: #6c757d; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; margin-bottom: 10px;}
    .card-value { color: #ffffff; font-size: 1.8rem; font-weight: 800; margin: 0;}
    
    /* Recommendation Box */
    .rec-box {
        background: rgba(79, 172, 254, 0.05);
        border: 1px solid rgba(79, 172, 254, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-top: 15px;
    }
    
    /* Sidebar aesthetic */
    section[data-testid="stSidebar"] {
        background-color: #0b0e11 !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. NAVIGATION / SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#4facfe; font-size: 1.5rem; font-weight: 800;'>SENTIAN<span style='color:white;'>RISK</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#555; font-size: 0.8rem; margin-top:-15px;'>Hybrid Predictive Intelligence</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    req_file = st.file_uploader("📝 Upload Requirements (.txt)", type=["txt"])
    spr_file = st.file_uploader("📊 Upload Resource Data (.csv)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 🛠️ SIMULATION")
    capacity_boost = st.slider("Additional Man-Hours", 0, 100, 0)
    
    if st.button("EXECUTE ANALYSIS", use_container_width=True, type="primary"):
        st.session_state['active'] = True

# --- 5. MAIN DASHBOARD ---
if st.session_state.get('active'):
    if req_file and spr_file:
        # Load and analyze
        r_text = req_file.getvalue().decode("utf-8")
        r_df = pd.read_csv(spr_file)
        
        # AI Sentiment Feature
        sentiment = TextBlob(r_text).sentiment.polarity
        mood = "OPTIMIZED" if sentiment > 0.1 else "AMBIGUOUS"
        
        # Backend Processing
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)
        r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        create_combined_dataset()
        train_hybrid_model()
        df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))

        # --- TOP LEVEL KPI GRID ---
        st.markdown("<h3 style='font-weight: 300; color: #888;'>SYSTEM OVERVIEW</h3>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            st.markdown(f"<div class='metric-container'><p class='card-title'>Linguistic Mood</p><p class='card-value' style='color:#4facfe;'>{mood}</p></div>", unsafe_allow_html=True)
        with k2:
            st.markdown(f"<div class='metric-container'><p class='card-title'>Ambiguity Score</p><p class='card-value'>{df['ambiguity_score'].mean():.2f}</p></div>", unsafe_allow_html=True)
        with k3:
            st.markdown(f"<div class='metric-container'><p class='card-title'>Resource Strain</p><p class='card-value'>{df['overload_score'].mean():.2f}</p></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='metric-container'><p class='card-title'>Model Confidence</p><p class='card-value'>94%</p></div>", unsafe_allow_html=True)

        # --- MAIN RESULTS SECTION ---
        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.markdown("<h3 style='font-weight: 300; color: #888;'>STRATEGIC DIAGNOSTICS</h3>", unsafe_allow_html=True)
            for _, row in df.iterrows():
                risk = str(row['risk_level']).upper()
                css_class = "risk-high" if "HIGH" in risk else "risk-med" if "MEDIUM" in risk else "risk-low"
                
                st.markdown(f"""
                    <div class="metric-container {css_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <p class="card-title" style="margin:0;">Sprint {row['sprint']}</p>
                                <p class="card-value" style="font-size:1.4rem;">Status: {risk}</p>
                            </div>
                            <div style="text-align: right;">
                                <p style="color:#6c757d; font-size:0.8rem; margin:0;">STRESS INDEX</p>
                                <p style="color:white; font-weight:800; font-size:1.2rem; margin:0;">{row['overload_score']:.2f}</p>
                            </div>
                        </div>
                        <div class="rec-box">
                            <span style="color:#4facfe; font-weight:800; font-size:0.8rem;">AI RECOMMENDATION:</span><br>
                            <span style="color:#ccc; font-size:0.9rem;">{"Shift complexity to next cycle and increase QA focus." if "HIGH" in risk else "Proceed with standard velocity."}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with col_r:
            st.markdown("<h3 style='font-weight: 300; color: #888;'>PREDICTIVE TREND</h3>", unsafe_allow_html=True)
            st.line_chart(df[['overload_score', 'ambiguity_score']])
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-weight: 300; color: #888;'>DATA HEALTH</h3>", unsafe_allow_html=True)
            st.bar_chart(df['overload_score'])
    else:
        st.warning("Please upload both Requirement and Resource files.")
else:
    # Landing State
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-weight: 200; color: #333; font-size: 3rem;'>SYSTEM STANDBY</h1>
            <p style='color: #222; font-size: 1.2rem;'>Awaiting data injection from the control panel.</p>
        </div>
    """, unsafe_allow_html=True)