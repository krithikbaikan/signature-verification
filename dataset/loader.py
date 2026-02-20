# dataset/loader.py

import os
from collections import defaultdict

def load_dataset_from_folder(folder_path, dataset_name="gpds"):
    data = defaultdict(lambda: {'genuine': [], 'forged': []})

    for label in ['genuine', 'forged']:
        dir_path = os.path.join(folder_path, label)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if not fname.lower().endswith(('.jpg', '.png')):
                continue
            if dataset_name == "gpds":
                user_id = fname.split('_')[0]
            elif dataset_name == "cedar":
                parts = fname.split('_')
                if len(parts) >= 2:
                    user_id = "cedar_" + parts[1]
                else:
                    print(f"[WARNING] Skipping malformed CEDAR filename: {fname}")
                    continue
            else:
                raise ValueError(f"Unknown dataset type: {dataset_name}")


            path = os.path.join(dir_path, fname)
            if label == 'genuine':
                data[user_id]['genuine'].append(path)
            else:
                data[user_id]['forged'].append(path)
    return data

def load_combined_datasets(base_path="data"):
    gpds = load_dataset_from_folder(os.path.join(base_path, "gpds"), "gpds")
    cedar = load_dataset_from_folder(os.path.join(base_path, "cedar"), "cedar")
    combined = gpds.copy()
    combined.update(cedar)
    return combined
