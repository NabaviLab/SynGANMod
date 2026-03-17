import torch.nn as nn

from .blocks import LearnablePositionalEncoding, PatchEmbedding, TransformerBlock


class MammogramTransformerEncoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        num_tokens = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(in_channels=1, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_embed = LearnablePositionalEncoding(num_tokens=num_tokens, embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        tokens, spatial_shape = self.patch_embed(x)
        tokens = self.pos_embed(tokens)
        tokens = self.blocks(tokens)
        return self.norm(tokens), spatial_shape
