import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_combined_data():
    file_path = os.path.join(PROJECT_ROOT, "results", "combined_risk_data.csv")
    return pd.read_csv(file_path)

def plot_risk_dashboard(df):
    sns.set(style="whitegrid")
    
    # Plot Ambiguity & Overload per sprint
    fig, ax1 = plt.subplots(figsize=(10,5))
    
    ax1.plot(df['sprint'], df['ambiguity_score'], marker='o', label='Ambiguity Score', color='blue')
    ax1.plot(df['sprint'], df['overload_score'], marker='o', label='Overload Score', color='green')
    ax1.set_xlabel("Sprint")
    ax1.set_ylabel("Score")
    ax1.set_title("Requirement Ambiguity & Sprint Overload per Sprint")
    ax1.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot risk level counts
    plt.figure(figsize=(6,4))
    sns.countplot(x='risk_level', data=df, palette='Reds')
    plt.title("Hybrid Risk Level Distribution")
    plt.show()
    
    # Print recommendations
    print("\n=== Recommendations ===")
    for i, row in df.iterrows():
        rec = "Monitor"
        if row['risk_level'] == "High":
            rec = "Immediate Action Required"
        elif row['risk_level'] == "Medium":
            rec = "Review Soon"
        print(f"Sprint {row['sprint']}: Risk={row['risk_level']} → {rec}")
