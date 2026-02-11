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
    st.error(f"Module import failed: {e}")

# --- 3. PAGE CONFIG & CSS ---
st.set_page_config(page_title="SentianRisk AI", layout="wide")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0E1117; }
    
    .metric-card {
        background: linear-gradient(145deg, #1e2129, #16181d);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 10px 10px 20px #080a0d;
    }
    
    .insight-box {
        background: rgba(79, 172, 254, 0.05);
        border-left: 4px solid #4facfe;
        padding: 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. HEADER ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h1 style='font-weight: 300;'>SENTIAN<span style='font-weight: 800; color: #4facfe;'>RISK</span> AI</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("Control Center")
    req_file = st.file_uploader("Upload Specs (.txt)", type=["txt"])
    spr_file = st.file_uploader("Upload Resources (.csv)", type=["csv"])
    if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
        st.session_state['active'] = True

# --- 6. MAIN CONTENT ---
if st.session_state.get('active'):
    if req_file and spr_file:
        # Save and Run
        os.makedirs(os.path.join(root_path, "data"), exist_ok=True)
        with open(os.path.join(root_path, "data", "requirements.txt"), "wb") as f: f.write(req_file.getvalue())
        pd.read_csv(spr_file).to_csv(os.path.join(root_path, "data", "sprint_tasks.csv"), index=False)

        create_combined_dataset()
        train_hybrid_model()

        df = pd.read_csv(os.path.join(root_path, "results", "combined_risk_data.csv"))

        # --- VISIBLE PROBLEM/SOLUTION SECTION ---
        st.subheader("💡 Strategic Insights & Problem Resolution")
        
        for _, row in df.iterrows():
            with st.container():
                risk = str(row['risk_level']).upper()
                clr = "#FF4B4B" if "HIGH" in risk else "#FFA500" if "MEDIUM" in risk else "#00FF00"
                
                # Dynamic Logic for Problem/Solution Visibility
                problem = ""
                solution = ""
                
                if "HIGH" in risk:
                    problem = f"Sprint {row['sprint']} exhibits critical resource overload ({row['overload_score']:.2f})."
                    solution = "Immediate reallocation of tasks or extension of sprint deadline recommended."
                elif "MEDIUM" in risk:
                    problem = f"Moderate linguistic ambiguity detected in requirements for Sprint {row['sprint']}."
                    solution = "Conduct a requirement refinement meeting to clarify vague technical specifications."
                else:
                    problem = "No significant risk factors detected."
                    solution = "Proceed with planned sprint velocity."

                st.markdown(f"""
                    <div class="insight-box">
                        <span style="color:{clr}; font-weight:bold;">SPRINT {row['sprint']} | {risk}</span><br>
                        <b style="color:#eee;">DETECTED PROBLEM:</b> <span style="color:#888;">{problem}</span><br>
                        <b style="color:#eee;">AI RECOMMENDATION:</b> <span style="color:#4facfe;">{solution}</span>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        # Visualizing the metrics below
        st.subheader("📊 Risk Distribution")
        st.line_chart(df[['ambiguity_score', 'overload_score']])
else:
    st.info("Upload files and click 'Run' to see the Problem/Solution mapping.")