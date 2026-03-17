from .default_config import DataConfig


def build_data_config(**overrides) -> DataConfig:
    cfg = DataConfig()
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
