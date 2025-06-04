import os
import pandas as pd

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_parquet(os.path.join(base_path, "dsr1_results_grpo.parquet"))

    unique_question = set()
    new_df = []
    for row in df.itertuples(index=False):
        if row.conversations[0]['from'] != 'user':
            continue

        question = row.conversations[0]['value']
        if question not in unique_question:
            new_df.append({
                "id": len(new_df),
                "label": row.label,
                "conversations": row.conversations
            })
            unique_question.add(row.conversations[0]['value'])
    
    df = pd.DataFrame(new_df)
    df.to_parquet(os.path.join(base_path, "dsr1_results_grpo_unique.parquet"))

    df = pd.read_parquet(os.path.join(base_path, "dsr1_results_grpo_unique.parquet"))