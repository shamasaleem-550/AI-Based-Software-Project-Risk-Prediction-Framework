import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_sprint_data():
    file_path = os.path.join(PROJECT_ROOT, "data", "sprint_tasks.csv")
    return pd.read_csv(file_path)

def compute_overload_metrics(df):
    results = []

    for sprint in df['sprint'].unique():
        sprint_df = df[df['sprint'] == sprint]
        total_tasks = len(sprint_df)
        incomplete_tasks = len(sprint_df[sprint_df['status'] != 'done'])
        carry_over_rate = incomplete_tasks / total_tasks

        tasks_per_dev = sprint_df.groupby('assignee').size().to_dict()
        max_tasks = max(tasks_per_dev.values())

        estimated_hours = sprint_df['estimated_hours'].sum()
        actual_hours = sprint_df['actual_hours'].sum()
        hours_ratio = actual_hours / max(estimated_hours, 1)

        overload_score = 0.4*carry_over_rate + 0.3*(max_tasks/10) + 0.3*hours_ratio
        overload_score = min(overload_score, 1.0)

        results.append({
            "sprint": sprint,
            "carry_over_rate": round(carry_over_rate,2),
            "max_tasks_per_dev": max_tasks,
            "hours_ratio": round(hours_ratio,2),
            "overload_score": round(overload_score,2)
        })

    return pd.DataFrame(results)

def save_overload_report(df):
    results_file = os.path.join(PROJECT_ROOT, "results", "overload_report.csv")
    df.to_csv(results_file, index=False)
    print(f"Overload report saved at {results_file}")

