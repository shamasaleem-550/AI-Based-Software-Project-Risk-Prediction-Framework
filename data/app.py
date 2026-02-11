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
    st.error(f"Module import failed. Check your folder structure. Error: {e}")

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Project Risk Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AI-Based Software Project Risk Predictor")
st.markdown("Developed by **SHAMA SALEEM** | Framework for NLP & Resource Risk Analysis")

# --- 4. SIDEBAR - FILE UPLOADS & SAMPLE DATA ---
with st.sidebar:
    st.header("📁 Data Input")
    st.write("Upload project files to begin.")
    
    requirements_file = st.file_uploader("1. Upload Requirements (.txt)", type=["txt"])
    sprint_file = st.file_uploader("2. Upload Sprint Tasks (.csv)", type=["csv"])
    
    st.markdown("---")
    st.subheader("💡 Don't have files?")
    st.caption("Use these templates to test the AI logic:")
    
    # Sample Text Data
    sample_req = "The system must be robust and fast. We might need a flexible login etc."
    st.download_button("📥 Download Sample .txt", sample_req, "sample_req.txt")

    # Sample CSV Data
    sample_csv = "sprint,task_name,hours_assigned,developer_capacity\n1,Database Setup,45,40\n2,UI Integration,20,40\n3,API Testing,50,40"
    st.download_button("📥 Download Sample .csv", sample_csv, "sample_sprint.csv")
    
    st.markdown("---")
    st.info("This tool detects ambiguity in text and overload in resource capacity.")

# --- 5. MAIN LOGIC ---
if requirements_file and sprint_file:
    if st.button("🔍 Run Full AI Risk Analysis", use_container_width=True):
        try:
            # Create directories
            os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
            os.makedirs(os.path.join(root_path, "results"), exist_ok=True)

            # Save files
            req_path = os.path.join(root_path, "data", "requirements.txt")
            with open(req_path, "wb") as f:
                f.write(requirements_file.getvalue())
            
            sprint_path = os.path.join(root_path, "data", "sprint_tasks.csv")
            with open(sprint_path, "wb") as f:
                f.write(sprint_file.getvalue())

            # AI Processing
            with st.spinner("🧠 AI Engine processing data..."):
                create_combined_dataset()
                train_hybrid_model()

            # --- 6. DISPLAY RESULTS ---
            res_path = os.path.join(root_path, "results", "combined_risk_data.csv")
            
            if os.path.exists(res_path):
                df = pd.read_csv(res_path)
                
                st.subheader("📊 Sprint Risk Dashboard")
                cols = st.columns(len(df))
                
                for i, (_, row) in enumerate(df.iterrows()):
                    with cols[i]:
                        risk = str(row['risk_level']).strip().upper()
                        color = "🔴" if risk == "HIGH" else "🟢"
                        st.metric(label=f"Sprint {row['sprint']}", value=f"{color} {risk}")

                # Charts
                st.markdown("### 📈 Technical Metrics")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Ambiguity Score (Text Analysis)**")
                    st.area_chart(df['ambiguity_score'])
                with c2:
                    st.write("**Overload Score (Resource Analysis)**")
                    st.bar_chart(df['overload_score'])

                # Download Report
                report_path = os.path.join(root_path, "results", "ambiguity_report.csv")
                if os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button("📥 Download Full Risk Report", f, "risk_report.csv", "text/csv")
            else:
                st.error("Processing failed. Results not found.")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
else:
    st.warning("👈 Please upload your Requirements and Sprint files in the sidebar to start.")