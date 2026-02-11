import pandas as pd
import os

def train_hybrid_model():
    """
    Advanced Heuristic Risk Engine:
    Calculates a weighted probability of project failure based on 
    Linguistic Ambiguity and Resource Strain.
    """
    results_path = "results/combined_risk_data.csv"
    
    if not os.path.exists(results_path):
        return "Combined dataset not found. Run combined_data.py first."

    df = pd.read_csv(results_path)

    # --- ADVANCED RISK CALCULATION ---
    # We assign weights: Resource Overload is high impact (0.7), Ambiguity is medium (0.3)
    W_RESOURCE = 0.7
    W_AMBIGUITY = 0.3

    def calculate_advanced_risk(row):
        # 1. Normalize scores to a 0-1 scale
        ambiguity = min(row['ambiguity_score'] / 10, 1.0)
        # Resource strain starts being a risk after 1.0 (100% capacity)
        resource_strain = max(0, (row['overload_score'] - 1.0)) 
        
        # 2. Weighted Score
        composite_score = (ambiguity * W_AMBIGUITY) + (resource_strain * W_RESOURCE)
        
        # 3. Decision Logic (Advanced Thresholds)
        if resource_strain > 0.4 or composite_score > 0.7:
            return "HIGH (Critical)"
        elif composite_score > 0.35:
            return "MEDIUM (Warning)"
        else:
            return "LOW (Stable)"

    df['risk_level'] = df.apply(calculate_advanced_risk, axis=1)
    
    # Save the sophisticated results
    df.to_csv(results_path, index=False)
    print("Advanced Hybrid Risk Analysis Complete.")

if __name__ == "__main__":
    train_hybrid_model()