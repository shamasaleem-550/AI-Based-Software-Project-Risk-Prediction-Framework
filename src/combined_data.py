import pandas as pd
import os

# Set project root folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_combined_dataset():
    # Load CSVs from results folder
    ambiguity_path = os.path.join(PROJECT_ROOT, "results", "ambiguity_report.csv")
    overload_path = os.path.join(PROJECT_ROOT, "results", "overload_report.csv")

    ambiguity = pd.read_csv(ambiguity_path)
    overload = pd.read_csv(overload_path)

    # Average ambiguity score per sprint
    avg_ambiguity = ambiguity['ambiguity_score'].mean()

    combined = pd.DataFrame()
    combined['sprint'] = overload['sprint']
    combined['ambiguity_score'] = avg_ambiguity
    combined['overload_score'] = overload['overload_score']

    # Synthetic risk labels
    def risk_label(row):
        if row['ambiguity_score'] > 0.5 or row['overload_score'] > 0.6:
            return 'High'
        elif row['ambiguity_score'] > 0.3 or row['overload_score'] > 0.4:
            return 'Medium'
        else:
            return 'Low'

    combined['risk_level'] = combined.apply(risk_label, axis=1)

    # Save combined dataset in results folder
    combined_path = os.path.join(PROJECT_ROOT, "results", "combined_risk_data.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Combined dataset saved at {combined_path}")

    return combined
