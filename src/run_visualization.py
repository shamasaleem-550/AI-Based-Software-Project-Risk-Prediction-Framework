import streamlit as st
import pandas as pd
import os

from src.combined_data import create_combined_dataset
from src.hybrid_risk_model import train_hybrid_model

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CUSTOM CLEAN STYLING
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("AI Risk Intelligence")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "About System"])

# -------------------------------
# ABOUT PAGE
# -------------------------------
if page == "About System":
    st.title("About This System")
    st.markdown("""
    **AI Risk Intelligence Platform**  

    This system predicts software project risk using:

    - Natural Language Processing (Requirement Ambiguity Detection)
    - Sprint Overload Analysis
    - Hybrid Machine Learning Model
    - Risk Classification & Visualization

    Designed for IT companies and software project managers.
    """)
    st.stop()

# -------------------------------
# DASHBOARD PAGE
# -------------------------------
st.title("📊 AI-Based Software Project Risk Intelligence")

st.markdown("Upload your project files to analyze risk levels using AI.")

st.markdown("---")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    requirements_file = st.file_uploader("Upload Requirements (.txt)", type=["txt"])

with col_upload2:
    sprint_file = st.file_uploader("Upload Sprint Tasks (.csv)", type=["csv"])

st.markdown("---")

if requirements_file and sprint_file:

    if st.button("Analyze Project Risk", use_container_width=True):

        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Save files
        with open("data/requirements.txt", "w") as f:
            f.write(requirements_file.getvalue().decode("utf-8"))

        sprint_df = pd.read_csv(sprint_file)
        sprint_df.to_csv("data/sprint_tasks.csv", index=False)

        with st.spinner("Running AI Risk Analysis..."):
            create_combined_dataset()
            train_hybrid_model()

        combined = pd.read_csv("results/combined_risk_data.csv")

        st.markdown("## 📈 Risk Dashboard")
        st.markdown("---")

        hig
