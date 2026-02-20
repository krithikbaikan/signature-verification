# models/encoder.py

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SignatureEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SignatureEncoder, self).__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.embedding = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)
