import torch
import torch.nn as nn


class IntensityConsistencyLoss(nn.Module):
    def forward(self, synthetic, current, blend_mask):
        return torch.mean(torch.abs((synthetic - current) * (1.0 - blend_mask)))
