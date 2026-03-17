"""Evaluation entry points for FID/FMD.

This file intentionally provides the structure for feature extraction and Fréchet
statistics computation. In practice, feature extractor selection can be swapped
between InceptionV3 and a medical backbone.
"""

import numpy as np


def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    return float(diff.dot(diff))


def compute_feature_stats(features: np.ndarray):
    return np.mean(features, axis=0), np.cov(features, rowvar=False)
