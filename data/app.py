import streamlit as st
import pandas as pd
from textblob import TextBlob # New: pip install textblob
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="SentianRisk AI | Elite", layout="wide")

# --- PREMIUM STYLING ---
st.markdown("""
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .sentiment-tag {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        background: #4facfe;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; font-weight: 800;'>SENTIAN<span style='color: #4facfe;'>RISK</span> PRO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d;'>AI-Powered Prescriptive Risk Modeling</p>", unsafe_allow_html=True)

# --- SIDEBAR: THE ADVANCED AI SIMULATOR ---
with st.sidebar:
    st.markdown("### 🛠️ SIMULATION ENGINE")
    st.caption("Adjust variables to see how AI suggests mitigation.")
    sim_capacity = st.slider("Bonus Capacity (Hours)", 0, 50, 0)
    sim_complexity = st.select_slider("Requirement Clarity", options=["Low", "Medium", "High"])
    
    st.markdown("---")
    uploaded_req = st.text_area("Enter Project Requirements", placeholder="E.g., The system must be fast and secure...")

# --- MAIN DASHBOARD LOGIC ---
if uploaded_req:
    # 1. AI Feature: Sentiment Analysis
    sentiment = TextBlob(uploaded_req).sentiment.polarity
    mood = "Confident" if sentiment > 0.2 else "Uncertain" if sentiment < 0 else "Neutral"
    
    # 2. Mock Data Calculation (Replacing with Sim Logic)
    # In a real app, this would call your hybrid_risk_model.py
    base_risk = 0.8 if mood == "Uncertain" else 0.4
    final_risk = max(0.1, base_risk - (sim_capacity/100)) # Simulation reduces risk
    
    # --- DASHBOARD UI ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
            <div class="glass-card">
                <p style="color:#888;">AI MOOD DETECTION</p>
                <h3>{mood}</h3>
                <span class="sentiment-tag">Score: {sentiment:.2f}</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        status_color = "#FF4B4B" if final_risk > 0.6 else "#00FF00"
        st.markdown(f"""
            <div class="glass-card">
                <p style="color:#888;">PREDICTED RISK LEVEL</p>
                <h2 style="color:{status_color};">{(final_risk*100):.0f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Strategic Recommendation")
        if final_risk > 0.6:
            st.error(f"⚠️ **PROBLEM DETECTED:** {mood} requirements + high workload.")
            st.write(f"**AI SOLUTION:** The simulation shows adding **{sim_capacity}hrs** is NOT enough. You must refine the requirement clarity to 'High' to drop risk below 50%.")
        else:
            st.success("✅ **OPTIMIZED:** Current configuration is stable.")
            st.write("The AI suggests proceeding with the current sprint velocity.")

    st.markdown("---")
    st.markdown("### 📊 Live Risk Heatmap")
    # Using a simple chart to represent the 2D Heatmap
    chart_data = pd.DataFrame({'Impact': [0.8, 0.4, 0.2], 'Probability': [final_risk, 0.3, 0.1]})
    st.scatter_chart(chart_data, x='Impact', y='Probability', size='Probability', color="#4facfe")

else:
    st.info("👈 Enter project requirements in the sidebar to initialize the AI Brain.")