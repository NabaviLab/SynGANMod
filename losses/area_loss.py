import torch
import torch.nn as nn


class AreaRegularizationLoss(nn.Module):
    def __init__(self, max_area_fraction: float = 0.12):
        super().__init__()
        self.max_area_fraction = max_area_fraction

    def forward(self, blend_mask):
        area = blend_mask.mean(dim=(1, 2, 3))
        return torch.mean(torch.relu(area - self.max_area_fraction) ** 2)
