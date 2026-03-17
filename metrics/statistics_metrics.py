import numpy as np
from scipy.stats import entropy, kurtosis, skew


def summarize_image(image: np.ndarray):
    flat = image.astype(np.float32).reshape(-1)
    hist, _ = np.histogram(flat, bins=256, range=(0.0, 1.0), density=True)
    return {
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "skewness": float(skew(flat)),
        "kurtosis": float(kurtosis(flat)),
        "entropy": float(entropy(hist + 1e-8)),
    }
