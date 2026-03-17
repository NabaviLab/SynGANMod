import torch.nn as nn


class TumorLocalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, tumor_map, tumor_mask):
        return self.loss(tumor_map, tumor_mask)
