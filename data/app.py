import sys
import os
import pandas as pd
import streamlit as st
from textblob import TextBlob
import nltk

# --- 1. CLOUD-SAFE NLTK SETUP ---
# This ensures the server downloads the necessary 'brain' for TextBlob
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('brown')
        nltk.download('punkt_tab')
    except Exception as e:
        st.error(f"NLTK Download failed: {e}")

download_nltk_data()

# --- 2. SYSTEM PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# --- 3. BACKEND IMPORTS ---
try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError as e:
    st.error(f"Backend modules not found. Ensure src/ exists. Error: {e}")

# --- 4. PAGE CONFIG & PREMIUM CSS ---
st.set_page_config(page_title="SentianRisk Pro", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0E1117; }
    .glass-card {
        background: linear-gradient(145deg, #1e2129, #16181d);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 35px;
        box-shadow: 10px 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    .status-pill {
        padding: 5px 15px; border-radius: 50px; font-size: 0.75rem; font-weight: 700;
        background: rgba(79, 172, 254, 0.1); color: #4facfe; border: 1px solid #4facfe;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 5. HEADER ---
st.markdown("<br>", unsafe_allow_html=True)
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("<h1 style='font-weight:300; margin:0;'>SENTIAN<span style='font-weight:800; color:#4facfe;'>RISK</span> PRO</h1>", unsafe_allow_html=True)
    st.markdown("<span class='status-pill'>NEURAL ANALYTICS ENGINE v3.0</span>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div style='text-align:right; border-right:3px solid #4facfe; padding-right:20px;'><span style='color:#4facfe; font-size:0.7rem; font-weight:800;'>EXECUTIVE LEAD</span><br><span style='color:white; font-size:1.1rem; font-weight:300;'>SHAMA SALEEM</span></div>", unsafe_allow_html=True)

st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.05); margin-top:30px; margin-bottom:40px;'>", unsafe_allow_html=True)

# --- 6. SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color:#4facfe;'>CONTROL PANEL</h3>", unsafe_allow_html=True)
    req_file = st.file_uploader("Upload Specs (.txt)", type=["txt"])
    spr_file = st.file_uploader("Upload Resources (.csv)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 🛠️ SIMULATION")
    bonus_hrs = st.slider("Additional Capacity", 0, 100, 0)
    
    if st.button("RUN DEEP ANALYSIS", type="primary", use_container_width=True):
        st.session_state['active'] = True

# --- 7. MAIN DASHBOARD ---
if st.session_state.get('active'):
    if req_file and spr_file:
        r_text = req_file.getvalue().decode("utf-8")
        r_df = pd.read_csv(spr_file)
        
        # AI Sentiment Feature
        sentiment = TextBlob(r_text).sentiment.polarity
        mood = "Stable/Clear" if sentiment > 0.1 else "Uncertain/Vague"
        
        # Backend Processing
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "w", encoding="utf-8") as f: 
            f.write(r_text)
        r_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        with st.spinner("AI Engine Simulating Scenarios..."):
            create_combined_dataset()
            train_hybrid_model()

        res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            
            i1, i2, i3 = st.columns(3)
            with i1: st.markdown(f"<div class='glass-card'><p style='color:#888;'>MOOD</p><h2>{mood}</h2></div>", unsafe_allow_html=True)
            with i2: st.markdown(f"<div class='glass-card'><p style='color:#888;'>AVG RISK</p><h2>{df['overload_score'].mean():.2f}</h2></div>", unsafe_allow_html=True)
            with i3: st.markdown(f"<div class='glass-card'><p style='color:#888;'>IMPACT</p><h2 style='color:#00FF00;'>-{bonus_hrs/10:.0f}%</h2></div>", unsafe_allow_html=True)

            st.markdown("### 🛰️ Risk Mitigation Strategy")
            for _, row in df.iterrows():
                risk_val = str(row['risk_level']).upper()
                clr = "#FF4B4B" if "HIGH" in risk_val else "#FFA500" if "MEDIUM" in risk_val else "#00FF00"
                st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.02); padding:20px; border-left:4px solid {clr}; border-radius:10px; margin-bottom:15px;">
                        <span style="font-weight:800; color:{clr};">SPRINT {row['sprint']} — {risk_val}</span><br>
                        <span style="color:#888;">AI Logic: {mood} Tone | Resource Stress: {row['overload_score']:.2f}</span>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please upload both files.")
else:
    st.markdown("<div style='text-align:center; padding:150px; color:#444;'><h2 style='font-weight:200;'>Neural Analysis Standby</h2></div>", unsafe_allow_html=True)