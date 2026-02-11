import streamlit as st
import pandas as pd
import os
import numpy as np

from src.combined_data import create_combined_dataset
from src.hybrid_risk_model import train_hybrid_model

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI Risk Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------
# CLEAN PROFESSIONAL STYLE
# ------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    font-weight: 600;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("AI Risk Intelligence")
menu = st.sidebar.radio("Navigation", ["Dashboard", "Architecture", "About"])

# ------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------
if menu == "About":
    st.title("About This AI System")

    st.markdown("""
    ### AI-Based Software Risk Intelligence Platform

    This system predicts software project risk using:

    - NLP-based Requirement Ambiguity Detection
    - Sprint Overload Analytics
    - Hybrid Machine Learning Model
    - Risk Classification & Visualization
    - Executive Reporting

    Designed for IT companies, project managers, and CTO dashboards.
    """)

    st.stop()

# ------------------------------------------------
# ARCHITECTURE PAGE
# ------------------------------------------------
if menu == "Architecture":
    st.title("System Architecture")

    st.markdown("""
    ### System Flow

    1. User uploads requirements & sprint tasks  
    2. Ambiguity detection module analyzes requirement clarity  
    3. Overload detection calculates sprint stress  
    4. Hybrid ML model predicts risk level  
    5. Dashboard visualizes executive insights  
    """)

    st.info("Architecture: Input → Feature Engineering → Hybrid Model → Risk Classification → Visualization")

    st.stop()

# ------------------------------------------------
# DASHBOARD
# ------------------------------------------------
st.title("📊 AI Software Project Risk Dashboard")

st.markdown("Upload project files for intelligent risk analysis.")

col1, col2 = st.columns(2)

with col1:
    req_file = st.file_uploader("Upload Requirements (.txt)", type=["txt"])

with col2:
    sprint_file = st.file_uploader("Upload Sprint Tasks (.csv)", type=["csv"])

if req_file and sprint_file:

    if st.button("Run AI Risk Analysis", use_container_width=True):

        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        with open("data/requirements.txt", "w") as f:
            f.write(req_file.getvalue().decode("utf-8"))

        sprint_df = pd.read_csv(sprint_file)
        sprint_df.to_csv("data/sprint_tasks.csv", index=False)

        with st.spinner("Running AI Model..."):
            create_combined_dataset()
            train_hybrid_model()

        combined = pd.read_csv("results/combined_risk_data.csv")

        # ------------------------------------------------
        # EXECUTIVE SUMMARY
        # ------------------------------------------------
        st.markdown("## Executive Summary")

        high = (combined["risk_level"] == "High").sum()
        medium = (combined["risk_level"] == "Medium").sum()
        low = (combined["risk_level"] == "Low").sum()

        total = len(combined)
        risk_score = round((high*1 + medium*0.5) / total * 100, 2)

        colA, colB, colC, colD = st.columns(4)

        colA.metric("Total Sprints", total)
        colB.metric("High Risk", high)
        colC.metric("Medium Risk", medium)
        colD.metric("Overall Risk %", f"{risk_score}%")

        if risk_score > 60:
            st.error("Project Status: CRITICAL 🚨")
        elif risk_score > 30:
            st.warning("Project Status: MODERATE ⚠")
        else:
            st.success("Project Status: STABLE ✅")

        st.markdown("---")

        # ------------------------------------------------
        # DETAILED SPRINT VIEW
        # ------------------------------------------------
        st.markdown("## Sprint-Level Risk Analysis")

        for _, row in combined.iterrows():
            st.markdown(f"### Sprint {row['sprint']}")
            colX
