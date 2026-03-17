from typing import Dict

import numpy as np
import torch


class ToTensorNormalize:
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict):
        output = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float()
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                output[key] = (tensor - self.mean) / self.std if key in {"prior", "current", "tumor_mask", "breast_mask"} else tensor
            else:
                output[key] = value
        return output
