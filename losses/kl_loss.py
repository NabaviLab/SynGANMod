import torch
import torch.nn as nn


class KLDivergenceLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
