"""Dark channel prior: dark channel, atmospheric light and raw transmission.

All functions take BGR images as produced by ``cv2.imread`` (uint8, 0-255)
unless stated otherwise.
"""

from __future__ import annotations

import cv2
import numpy as np


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """J_dark(x) = min_c min_{y in Omega(x)} J^c(y).

    Args:
        image: (H, W, 3) array, any numeric dtype.
        patch_size: side of the square local patch Omega (paper: 15).

    Returns:
        (H, W) float32 array, same value range as ``image``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {image.shape}")
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")

    min_channel = np.min(image.astype(np.float32), axis=2)
    # A grey-level erosion with a rectangular kernel is a local minimum filter.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def atmospheric_light(
    image: np.ndarray, dark: np.ndarray, top_fraction: float = 0.001
) -> np.ndarray:
    """Estimate the global atmospheric light A.

    Among the ``top_fraction`` pixels with the brightest dark channel (the most
    haze-opaque ones), pick the pixel with the highest intensity in the input.

    Returns:
        (3,) float64 array in the input's channel order (BGR).
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")

    h, w = dark.shape
    n_top = max(int(h * w * top_fraction), 1)

    flat_dark = dark.ravel()
    flat_img = image.reshape(-1, 3).astype(np.float64)

    # O(N) selection of the n_top largest values, no full sort needed.
    top_idx = np.argpartition(flat_dark, -n_top)[-n_top:]
    candidates = flat_img[top_idx]
    return candidates[np.argmax(candidates.sum(axis=1))]


def raw_transmission(
    image: np.ndarray, A: np.ndarray, patch_size: int = 15, omega: float = 0.95
) -> np.ndarray:
    """t~(x) = 1 - omega * dark_channel(I / A), clipped to [0, 1].

    ``omega < 1`` deliberately keeps a little haze for distant objects
    (aerial perspective); the paper uses 0.95.
    """
    A_safe = np.maximum(np.asarray(A, dtype=np.float64), 1e-6)
    normalized = image.astype(np.float64) / A_safe
    dark = dark_channel(normalized.astype(np.float32), patch_size)
    return np.clip(1.0 - omega * dark.astype(np.float64), 0.0, 1.0)
