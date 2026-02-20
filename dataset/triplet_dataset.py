# dataset/triplet_dataset.py

import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class TripletSignatureDataset(Dataset):
    def __init__(self, user_data, transform=None, hard_negative=True):
        self.user_data = user_data
        self.users = list(user_data.keys())
        self.transform = transform or transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(0.2, 0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.hard_negative = hard_negative
        self.triplets = self._generate_triplets()

    def _generate_triplets(self):
        triplets = []
        for user, samples in self.user_data.items():
            genuine = samples['genuine']
            forged = samples['forged']
            if len(genuine) < 2 or len(forged) < 1:
                continue

            for i in range(len(genuine) - 1):
                anchor = genuine[i]
                positive = genuine[i + 1]
                
                if self.hard_negative:
                    # Sample multiple forged examples to increase difficulty
                    negatives = random.sample(forged, min(3, len(forged)))
                    for negative in negatives:
                        triplets.append((anchor, positive, negative))
                else:
                    negative = random.choice(forged)
                    triplets.append((anchor, positive, negative))
        print(f"[INFO] Total triplets generated: {len(triplets)}")
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]
        a = self.transform(Image.open(a_path).convert("L"))
        p = self.transform(Image.open(p_path).convert("L"))
        n = self.transform(Image.open(n_path).convert("L"))
        return a, p, n
