import pandas as pd

def train_hybrid_model():

    df = pd.read_csv("results/combined_risk_data.csv")

    # Hybrid scoring logic (rule-based ML style)
    def classify(row):
        score = (row["ambiguity_score"] * 0.6) + (row["overload_score"] * 0.4)

        if score > 0.7:
            return "High"
        elif score > 0.4:
            return "Medium"
        else:
            return "Low"

    df["risk_level"] = df.apply(classify, axis=1)

    df.to_csv("results/combined_risk_data.csv", index=False)

    return df
