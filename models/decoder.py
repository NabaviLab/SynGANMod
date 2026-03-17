import torch
import torch.nn as nn

from .blocks import CrossAttentionBlock, TransformerBlock


class TumorTransformerDecoder(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int, latent_dim: int, depth: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_tokens = self.grid_size ** 2
        self.embed_dim = embed_dim

        self.latent_to_tokens = nn.Linear(latent_dim, self.num_tokens * embed_dim)
        self.self_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(max(depth - 1, 1))
        ])
        self.cross_block = CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.to_patch_logits = nn.Conv2d(embed_dim, 1, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False)

    def forward(self, z: torch.Tensor, fused_tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.latent_to_tokens(z).view(z.size(0), self.num_tokens, self.embed_dim)
        tokens = self.self_blocks(tokens)
        tokens = self.cross_block(tokens, fused_tokens)
        tokens = self.norm(tokens)
        tokens = tokens.transpose(1, 2).reshape(z.size(0), self.embed_dim, self.grid_size, self.grid_size)
        tumor_map = self.to_patch_logits(tokens)
        tumor_map = self.upsample(tumor_map)
        return torch.sigmoid(tumor_map)
