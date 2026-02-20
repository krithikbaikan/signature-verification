# generate_eval_pairs.py

import os
import csv
from dataset.loader import load_combined_datasets

def generate_eval_pairs(base_path="data", output_file="data/eval_pairs.csv", pairs_per_user=5):
    user_data = load_combined_datasets(base_path)
    pairs = []

    for user, samples in user_data.items():
        genuines = samples['genuine']
        forgeries = samples['forged']

        # Genuine pairs
        if len(genuines) >= 2:
            for i in range(min(pairs_per_user, len(genuines) - 1)):
                pairs.append([genuines[i], genuines[i+1], 1])

        # Forged pairs
        if len(genuines) >= 1 and len(forgeries) >= 1:
            for i in range(min(pairs_per_user, len(forgeries))):
                pairs.append([genuines[0], forgeries[i], 0])

    print(f"[INFO] Generated {len(pairs)} evaluation pairs.")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['anchor', 'pair', 'label'])
        writer.writerows(pairs)

    print(f"[INFO] Saved to {output_file}")

if __name__ == "__main__":
    generate_eval_pairs()
