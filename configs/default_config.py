from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DataConfig:
    image_size: int = 1024
    patch_size: int = 16
    num_workers: int = 4
    batch_size: int = 4
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


@dataclass
class ModelConfig:
    embed_dim: int = 256
    latent_dim: int = 256
    encoder_depth: int = 6
    decoder_depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    swin_name: str = "swin_tiny_patch4_window7_224"
    swin_pretrained: bool = True
    max_area_fraction: float = 0.12


@dataclass
class TrainConfig:
    epochs: int = 200
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 1e-4
    weight_decay: float = 1e-5
    beta1: float = 0.5
    beta2: float = 0.999
    amp: bool = True
    grad_clip: float = 1.0
    save_every: int = 5
    log_every: int = 20
    lambda_kl: float = 0.1
    lambda_adv: float = 1.0
    lambda_tumor: float = 5.0
    lambda_intensity: float = 10.0
    lambda_area: float = 2.0


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict:
        return {
            "seed": self.seed,
            "data": vars(self.data),
            "model": vars(self.model),
            "train": vars(self.train),
        }
