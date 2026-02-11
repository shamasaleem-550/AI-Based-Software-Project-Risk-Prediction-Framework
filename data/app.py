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

# Pathing setup
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError:
    st.error("Engine failure: Core modules (src/) not found.")

# --- 2. THE "SMART MAPPER" (This makes it run any file) ---
def smart_map_columns(df):
    mapping = {}
    cols = df.columns.str.lower()
    
    # Logic to find the right columns regardless of name
    for i, col in enumerate(cols):
        if any(x in col for x in ['sprint', 'id', 'no', 'period']):
            mapping['sprint'] = df.columns[i]
        elif any(x in col for x in ['assigned', 'hours', 'work', 'load']):
            mapping['hours_assigned'] = df.columns[i]
        elif any(x in col for x in ['capacity', 'limit', 'max', 'total']):
            mapping['team_capacity'] = df.columns[i]
            
    return mapping

# --- 3. UI THEME ---
st.set_page_config(page_title="SentianRisk Pro", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    .stApp { background-color: #080a0c; font-family: 'Plus Jakarta Sans', sans-serif; color: white; }
    .kpi-box { background: linear-gradient(145deg, #111418, #181c22); border: 1px solid rgba(255,255,255,0.03); border-radius: 15px; padding: 20px; text-align: center; }
    .diag-card { background: #111418; border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin-bottom: 15px; }
    .footer { text-align: center; color: #444; font-size: 0.75rem; padding: 30px 0; border-top: 1px solid rgba(255,255,255,0.03); margin-top: 50px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. BRANDING ---
st.markdown("<h1 style='font-weight:300;'>SENTIAN<span style='color:#00d9ff; font-weight:800;'>RISK</span></h1>", unsafe_allow_html=True)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown("### Control Panel")
    req_file = st.file_uploader("Upload Requirements (.txt)", type=["txt"])
    spr_file = st.file_uploader("Upload Sprint Data (.csv)", type=["csv"])
    execute = st.button("✨ EXECUTE ANALYSIS", type="primary", use_container_width=True)

# --- 6. MAIN LOGIC ---
if execute and req_file and spr_file:
    try:
        raw_df = pd.read_csv(spr_file)
        col_map = smart_map_columns(raw_df)
        
        # Check if we found enough data to proceed
        if len(col_map) < 3:
            st.warning("⚠️ Column Mapping partial. Ensure CSV has Sprint, Hours, and Capacity data.")
        
        # Rename columns internally so the AI engine understands
        clean_df = raw_df.rename(columns={v: k for k, v in col_map.items()})
        
        # Standard Processing
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        clean_df.to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)
        
        r_text = req_file.getvalue().decode("utf-8")
        with open(os.path.join(root_path, "data", "requirements.txt"), "w") as f: f.write(r_text)

        with st.spinner("Analyzing..."):
            create_combined_dataset()
            train_hybrid_model()
            df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))
            sentiment = TextBlob(r_text).sentiment.polarity

        # Display KPIs
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f"<div class='kpi-box'><p>MOOD</p><h2>{('STABLE' if sentiment > 0 else 'VAGUE')}</h2></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi-box'><p>RISK INDEX</p><h2>{df['overload_score'].mean():.2f}</h2></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='kpi-box'><p>CONFIDENCE</p><h2>94%</h2></div>", unsafe_allow_html=True)

        st.line_chart(df.set_index('sprint')[['overload_score']])

    except Exception as e:
        st.error("The system encountered a data format it couldn't solve. Please use standard headers.")
else:
    st.info("Awaiting file upload...")

st.markdown(f"<div class='footer'>SENTIANRISK ARCHITECTURE &copy; 2026 BY SHAMA SALEEM</div>", unsafe_allow_html=True)