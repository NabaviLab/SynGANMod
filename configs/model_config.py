from .default_config import ModelConfig


def build_model_config(**overrides) -> ModelConfig:
    cfg = ModelConfig()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
