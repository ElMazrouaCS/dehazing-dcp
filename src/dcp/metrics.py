"""Full-reference image quality metrics (PSNR, SSIM).

Both images must have the same shape: nothing is resized silently.
"""

from __future__ import annotations

import cv2
import numpy as np


def _check(img1: np.ndarray, img2: np.ndarray) -> None:
    if img1.shape != img2.shape:
        raise ValueError(f"shape mismatch: {img1.shape} vs {img2.shape}")


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB (inf for identical images)."""
    _check(img1, img2)
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float("inf")
    return float(10.0 * np.log10(max_val**2 / mse))


def ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    win_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    max_val: float = 255.0,
) -> float:
    """Mean SSIM (Wang et al., 2004) with an 11x11 Gaussian window, sigma = 1.5.

    Colour images: SSIM is computed per channel and averaged. Border pixels
    whose window would leave the image are excluded from the mean, which is the
    convention of ``skimage.metrics.structural_similarity`` (checked in tests).
    """
    _check(img1, img2)
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (win_size, win_size), sigma)

    mu1, mu2 = blur(i1), blur(i2)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = blur(i1 * i1) - mu1_sq
    s2 = blur(i2 * i2) - mu2_sq
    s12 = blur(i1 * i2) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    pad = win_size // 2
    return float(ssim_map[pad:-pad, pad:-pad].mean())


def evaluate_pair(output: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {"psnr": psnr(output, reference), "ssim": ssim(output, reference)}
