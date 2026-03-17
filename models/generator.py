import torch.nn as nn

from .blending import AnatomicalBlendingModule
from .decoder import TumorTransformerDecoder
from .encoder import MammogramTransformerEncoder
from .fusion import TemporalFusion, ViewSideEmbedding
from .latent import VariationalLatentUnit


class ProjectionAwareTumorGenerator(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int, latent_dim: int, encoder_depth: int, decoder_depth: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.num_tokens = (image_size // patch_size) ** 2
        self.prior_encoder = MammogramTransformerEncoder(image_size, patch_size, embed_dim, encoder_depth, num_heads, mlp_ratio, dropout)
        self.current_encoder = MammogramTransformerEncoder(image_size, patch_size, embed_dim, encoder_depth, num_heads, mlp_ratio, dropout)
        self.geom_embed = ViewSideEmbedding(embed_dim)
        self.fusion = TemporalFusion(embed_dim, num_heads, mlp_ratio, dropout)
        self.latent = VariationalLatentUnit(self.num_tokens, embed_dim, latent_dim)
        self.decoder = TumorTransformerDecoder(image_size, patch_size, embed_dim, latent_dim, decoder_depth, num_heads, mlp_ratio, dropout)
        self.blending = AnatomicalBlendingModule(latent_dim)

    def forward(self, prior, current, breast_mask, view_ids, side_ids):
        prior_tokens, _ = self.prior_encoder(prior)
        current_tokens, _ = self.current_encoder(current)
        prior_tokens = self.geom_embed(prior_tokens, view_ids, side_ids)
        current_tokens = self.geom_embed(current_tokens, view_ids, side_ids)
        fused_tokens = self.fusion(current_tokens, prior_tokens)
        z, mu, logvar = self.latent(fused_tokens)
        tumor_map = self.decoder(z, fused_tokens)
        synthetic, blend_mask, alpha = self.blending(current, tumor_map, breast_mask, z)
        return {
            "synthetic": synthetic,
            "tumor_map": tumor_map,
            "blend_mask": blend_mask,
            "alpha": alpha,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "fused_tokens": fused_tokens,
        }
