"""

# evaluation/evaluate.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from models.encoder import SignatureEncoder

def load_image(path):
    img = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def evaluate(model_path="outputs/models/final_model.pt", pair_file="data/eval_pairs.csv"):
    print("[INFO] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("[INFO] Loading pairs...")
    df = pd.read_csv(pair_file)
    if df.empty:
        print("[ERROR] eval_pairs.csv is empty!")
        return

    distances, labels = [], []

    for idx, row in df.iterrows():
        try:
            a = load_image(row['anchor']).to(device)
            p = load_image(row['pair']).to(device)
            label = int(row['label'])

            with torch.no_grad():
                a_emb = model(a)
                p_emb = model(p)
                dist = F.pairwise_distance(a_emb, p_emb).item()

            distances.append(dist)
            labels.append(label)
        except Exception as e:
            print(f"[SKIP] {row['anchor']}, {row['pair']} → {e}")

    distances = np.array(distances)
    labels = np.array(labels)

    if len(distances) == 0:
        print("[ERROR] No valid image pairs evaluated.")
        return

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    eer_thresh = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    accuracy = np.mean((distances < -eer_thresh).astype(int) == labels)
    roc_auc = auc(fpr, tpr)

    print(f"[RESULT] AUC: {roc_auc:.4f}")
    print(f"[RESULT] EER: {eer:.4f}")
    print(f"[RESULT] Best Threshold: {-eer_thresh:.4f}")
    print(f"[RESULT] Accuracy @ EER Threshold: {accuracy*100:.2f}%")

if __name__ == "__main__":
    evaluate()

    """
# evaluation/evaluate.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from PIL import Image

from models.encoder import SignatureEncoder

def load_image(path):
    image = Image.open(path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def evaluate(model_path="outputs/models/final_model.pt", pair_file="data/eval_pairs.csv"):
    print("[INFO] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignatureEncoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.read_csv(pair_file)
    distances, labels = [], []

    for _, row in df.iterrows():
        try:
            a = load_image(row['anchor']).to(device)
            p = load_image(row['pair']).to(device)
            label = int(row['label'])
            with torch.no_grad():
                a_emb = model(a)
                p_emb = model(p)
                dist = F.pairwise_distance(a_emb, p_emb).item()
            distances.append(dist)
            labels.append(label)
        except Exception as e:
            print(f"[SKIP] Error: {e}")

    distances = np.array(distances)
    labels = np.array(labels)

    # ROC & EER
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    best_thresh = thresholds[np.nanargmin(np.absolute(fnr - fpr))]

    # Accuracy at best threshold
    preds = (distances < -best_thresh).astype(int)
    acc = np.mean(preds == labels)

    print(f"[RESULT] AUC: {roc_auc:.4f}")
    print(f"[RESULT] EER: {eer:.4f}")
    print(f"[RESULT] Best Threshold: {-best_thresh:.4f}")
    print(f"[RESULT] Accuracy @ Best Threshold: {acc*100:.2f}%")

    # 📊 Histogram
    plt.figure(figsize=(6,4))
    plt.hist([distances[i] for i in range(len(labels)) if labels[i] == 1], bins=40, alpha=0.6, label='Genuine')
    plt.hist([distances[i] for i in range(len(labels)) if labels[i] == 0], bins=40, alpha=0.6, label='Forged')
    plt.axvline(x=-best_thresh, color='red', linestyle='--', label='Threshold')
    plt.title("Distance Histogram")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # 📈 ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
