from typing import Tuple

import cv2
import numpy as np


def read_grayscale(path: str, image_size: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def apply_clahe(image: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_u8)
    return enhanced.astype(np.float32) / 255.0


def binary_mask(path: str, image_size: int) -> np.ndarray:
    mask = read_grayscale(path, image_size)
    return (mask > 0.5).astype(np.float32)
