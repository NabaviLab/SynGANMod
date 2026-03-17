import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def d_loss(self, real_pred, fake_pred):
        real_target = torch.ones_like(real_pred)
        fake_target = torch.zeros_like(fake_pred)
        return self.bce(real_pred, real_target) + self.bce(fake_pred, fake_target)

    def g_loss(self, fake_pred):
        target = torch.ones_like(fake_pred)
        return self.bce(fake_pred, target)
