# encoder.py
import torch
import torch.nn as nn
from torchvision.models import resnet50

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.resnet = resnet50(weights="IMAGENET1K_V1")
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.fc = nn.Linear(2048, embed_size)

    def forward(self, images):
        features = self.resnet(images)  # Shape: (batch_size, 2048, 14, 14)
        features = features.permute(0, 2, 3, 1)  # Shape: (batch_size, 14, 14, 2048)
        features = features.view(features.size(0), -1, features.size(3))  # Shape: (batch_size, num_pixels, 2048)
        return features
