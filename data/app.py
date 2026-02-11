import sys
import os
import pandas as pd
import streamlit as st

# --- 1. SYSTEM PATH SETUP ---
# This ensures the app can find your 'src' folder whether running locally or on the web
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(current_dir, '..'))

if root_path not in sys.path:
    sys.path.insert(0, root_path)

# --- 2. BACKEND IMPORTS ---
try:
    from src.combined_data import create_combined_dataset
    from src.hybrid_risk_model import train_hybrid_model
except ImportError as e:
    st.error(f"Module import failed. Please check your folder structure. Error: {e}")

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Project Risk Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI-Based Software Project Risk Predictor")
st.markdown("Predicting software project success using NLP and Resource Capacity Analysis.")

# --- 4. SIDEBAR - FILE UPLOADS ---
with st.sidebar:
    st.header("📁 Data Input")
    st.info("Upload your project files to begin analysis.")
    requirements_file = st.file_uploader("Upload Requirements (.txt)", type=["txt"])
    sprint_file = st.file_uploader("Upload Sprint Tasks (.csv)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 🛠️ Tool Info")
    st.write("This tool uses a Hybrid Machine Learning model to detect ambiguity in text and overload in task assignments.")

# --- 5. MAIN LOGIC & ANALYSIS ---
if requirements_file and sprint_file:
    if st.button("🔍 Run Full AI Risk Analysis", use_container_width=True):
        try:
            # Create necessary directories
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(root_path, "results"), exist_ok=True)

            # Save uploaded files to the 'data' folder
            req_path = os.path.join(root_path, "data", "requirements.txt")
            with open(req_path, "wb") as f:
                f.write(requirements_file.getvalue())
            
            sprint_path = os.path.join(root_path, "data", "sprint_tasks.csv")
            with open(sprint_path, "wb") as f:
                f.write(sprint_file.getvalue())

            # Run AI Engine
            with st.spinner("🧠 AI is processing requirements and training risk models..."):
                create_combined_dataset()
                train_hybrid_model()

            # --- 6. DISPLAY RESULTS ---
            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            
            if os.path.exists(res_path):
                combined_df = pd.read_csv(res_path)
                
                st.markdown("---")
                st.subheader("📊 Executive Risk Summary")
                
                # Metric Columns for Sprints
                cols = st.columns(len(combined_df))
                for i, (_, row) in enumerate(combined_df.iterrows()):
                    with cols[i]:
                        risk = row['risk_level'].strip().capitalize()
                        if risk == "High":
                            st.metric(label=f"Sprint {row['sprint']}", value="🔴 HIGH RISK")
                        else:
                            st.metric(label=f"Sprint {row['sprint']}", value="🟢 LOW RISK")

                # Charts Section
                st.markdown("### 📈 Risk Factor Breakdown")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.write("**Ambiguity Score (NLP)**")
                    st.bar_chart(combined_df[['ambiguity_score']])
                
                with chart_col2:
                    st.write("**Overload Score (Resource)**")
                    st.bar_chart(combined_df[['overload_score']])

                # --- 7. DOWNLOAD SECTION ---
                st.markdown("---")
                report_path = os.path.join(root_path, "results", "ambiguity_report.csv")
                if os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Detailed Risk Report (CSV)",
                            data=f,
                            file_name="risk_analysis_report.csv",
                            mime="text/csv",
                            use_container_width=True 
                        )
                st.success("Analysis Complete! Actionable insights generated.")
            else:
                st.error("Error: Could not find result files. Check the 'results' folder.")

        except Exception as e:
            st.error(f"Critical Error during analysis: {e}")
            st.info("Check if your CSV columns match the required format: 'sprint', 'task_name', 'hours_assigned', 'developer_capacity'")

else:
    # Landing page state
    st.warning("👈 Please upload both a Requirements file and a Sprint Task file in the sidebar to start.")
    
    # Simple instructions for the user
    with st.expander("❓ How to use this tool"):
        st.write("""
        1. **Upload Requirements:** A .txt file containing the project scope.
        2. **Upload Sprint Tasks:** A .csv file with columns: `sprint`, `task_name`, `hours_assigned`, and `developer_capacity`.
        3. **Analyze:** Click the button to see AI-generated risk levels.
        """)