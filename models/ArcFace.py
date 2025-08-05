import torch
import torch.nn  as nn

class ArcFace(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        W = F.normalize(self.weight)

        # Cosine similarity
        cos_theta = torch.matmul(embeddings, W.t()).clamp(-1, 1)

        # Apply angular margin
        theta = torch.acos(cos_theta)
        cos_theta_m = torch.cos(theta + self.m)

        # One-hot encoding
        one_hot = F.one_hot(labels, num_classes=num_classes).float()

        # Apply margin to correct class
        logits = self.s * (one_hot * cos_theta_m + (1 - one_hot) * cos_theta)
        return logits