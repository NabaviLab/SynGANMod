import torch
import torch.nn as nn


class AnatomicalBlendingModule(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.mask_refine = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha_head = nn.Linear(latent_dim, 1)

    def forward(self, current: torch.Tensor, tumor_map: torch.Tensor, breast_mask: torch.Tensor, z: torch.Tensor):
        blend_mask = self.mask_refine(torch.cat([tumor_map, breast_mask], dim=1))
        constrained_mask = blend_mask * breast_mask
        alpha = 0.8 + 0.2 * torch.sigmoid(self.alpha_head(z)).view(-1, 1, 1, 1)
        synthetic = (1.0 - constrained_mask) * current + constrained_mask * (alpha * tumor_map)
        return synthetic, constrained_mask, alpha
