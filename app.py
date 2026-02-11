import streamlit as st

# Page config
st.set_page_config(
    page_title="AI Software Project Risk Predictor",
    layout="centered"
)

# App title
st.title("🚀 AI-Based Software Project Risk Predictor")

# Description
st.write("""
This web app analyzes software project risk using:
- Requirement ambiguity analysis (NLP)
- Sprint task overload detection
- Hybrid AI risk prediction

Upload project data to get risk insights.
""")

st.success("✅ App loaded successfully!")
