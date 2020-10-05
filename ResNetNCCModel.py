from torchvision import models
from torch import nn


class GenNCC(nn.Module):
    def __init__(self):
        super(GenNCC, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 512))
        self.relu = nn.ReLU()
        self.logits = nn.Linear(512, 20)
        self.prob = nn.Sigmoid()

    def forward(self, features):
        feature_scores = self.layers(features)
        r = self.relu(feature_scores)
        logits = self.logits(r)
        prob = self.prob(logits)
        return feature_scores, logits, prob
