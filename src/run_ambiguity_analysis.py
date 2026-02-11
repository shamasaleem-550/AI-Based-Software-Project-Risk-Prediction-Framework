import pandas as pd
from requirement_analysis import analyze_requirement, compute_ambiguity_score
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_file = os.path.join(PROJECT_ROOT, "data", "requirements.txt")
results_file = os.path.join(PROJECT_ROOT, "results", "ambiguity_report.csv")

with open(data_file, "r") as f:
    requirements = f.readlines()

results = []
for req in requirements:
    metrics = analyze_requirement(req)
    score = compute_ambiguity_score(metrics)
    results.append({
        "requirement": req.strip(),
        **metrics,
        "ambiguity_score": round(score, 2)
    })

df = pd.DataFrame(results)
df.to_csv(results_file, index=False)
print(f"Ambiguity report generated at {results_file}")
