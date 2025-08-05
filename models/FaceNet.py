import torch
import torch.nn as nn
from models.EfficientFRBackbone import efficientFRBackbone
from models.ArcFace import ArcFace


class FaceNet(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        self.backbone = efficientFRBackbone(embedding_dim)
        self.arcface = ArcFace(embedding_dim, num_classes)

    def forward(self, x, labels=None):
        embeddings = self.backbone(x)
        if labels is not None:
            return self.arcface(embeddings, labels)
        return embeddings