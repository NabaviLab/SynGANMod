import torch
import torch.nn as nn


class VariationalLatentUnit(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int, latent_dim: int):
        super().__init__()
        flattened_dim = num_tokens * embed_dim
        self.mu_head = nn.Linear(flattened_dim, latent_dim)
        self.logvar_head = nn.Linear(flattened_dim, latent_dim)

    def forward(self, fused_tokens: torch.Tensor):
        flat = fused_tokens.flatten(1)
        mu = self.mu_head(flat)
        logvar = self.logvar_head(flat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
