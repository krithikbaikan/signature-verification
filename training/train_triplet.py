# training/train_triplet.py

import sys, os, yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.loader import load_combined_datasets
from dataset.triplet_dataset import TripletSignatureDataset
from models.encoder import SignatureEncoder

def train_triplet_model(config_path='train_config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Loading data...")
    user_data = load_combined_datasets("data")
    dataset = TripletSignatureDataset(user_data, hard_negative=cfg['hard_negative'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

    print("[INFO] Building model...")
    model = SignatureEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = nn.TripletMarginLoss(margin=cfg['margin'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) if cfg['use_scheduler'] else None
    os.makedirs("outputs/models", exist_ok=True)

    print("[INFO] Starting training...")
    model.train()
    for epoch in range(cfg['epochs']):
        total_loss = 0
        for batch in dataloader:
            a, p, n = [x.to(device) for x in batch]
            anchor = model(a)
            positive = model(p)
            negative = model(n)

            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"outputs/models/epoch_{epoch+1}.pt")
        if scheduler: scheduler.step()

    torch.save(model.state_dict(), "outputs/models/final_model.pt")
    print("[INFO] Training complete.")

if __name__ == "__main__":
    train_triplet_model()
