import torch
import torch.nn as nn

from .blocks import CrossAttentionBlock


class ViewSideEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.view_embed = nn.Embedding(2, embed_dim)
        self.side_embed = nn.Embedding(2, embed_dim)

    def forward(self, tokens: torch.Tensor, view_ids: torch.Tensor, side_ids: torch.Tensor) -> torch.Tensor:
        geom = self.view_embed(view_ids) + self.side_embed(side_ids)
        return tokens + geom.unsqueeze(1)


class TemporalFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)

    def forward(self, current_tokens: torch.Tensor, prior_tokens: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(current_tokens, prior_tokens)
