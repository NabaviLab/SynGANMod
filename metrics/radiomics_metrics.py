import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_feature_similarity(real_features: np.ndarray, synthetic_features: np.ndarray) -> float:
    return float(cosine_similarity(real_features[None], synthetic_features[None])[0, 0])
