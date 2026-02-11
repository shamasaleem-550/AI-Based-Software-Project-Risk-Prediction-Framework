import pandas as pd
import os

def create_combined_dataset():

    # Read requirements text
    with open("data/requirements.txt", "r", encoding="utf-8") as f:
        text = f.read()

    sentences = [s.strip() for s in text.split(".") if s.strip()]

    # Simple ambiguity scoring (basic NLP logic)
    ambiguous_words = [
        "fast", "quick", "user-friendly", "efficient",
        "optimize", "improve", "secure", "robust"
    ]

    ambiguity_score = 0
    for sentence in sentences:
        for word in ambiguous_words:
            if word in sentence.lower():
                ambiguity_score += 1

    if len(sentences) > 0:
        ambiguity_score = ambiguity_score / len(sentences)
    else:
        ambiguity_score = 0

    # Read sprint CSV
    sprint_df = pd.read_csv("data/sprint_tasks.csv")

    # Overload = tasks per sprint normalized
    sprint_counts = sprint_df.groupby("sprint").size().reset_index(name="task_count")

    max_tasks = sprint_counts["task_count"].max()
    sprint_counts["overload_score"] = sprint_counts["task_count"] / max_tasks

    # Add ambiguity score to each sprint
    sprint_counts["ambiguity_score"] = ambiguity_score

    os.makedirs("results", exist_ok=True)
    sprint_counts.to_csv("results/combined_risk_data.csv", index=False)

    return sprint_counts
