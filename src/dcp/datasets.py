"""Pairing hazy / clean images of paired dehazing datasets.

Supported naming conventions (the ID is the part before the first "_"):
  O-HAZE / I-HAZE / Dense-HAZE: 01_outdoor_hazy.jpg  <->  01_outdoor_GT.jpg
  RESIDE SOTS:                  0001_0.8_0.2.jpg     <->  0001.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_id(path: Path) -> str:
    return path.stem.split("_")[0]


def pair_images(hazy_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    """Return (hazy, gt) pairs sorted by hazy file name.

    Several hazy images may share one GT (e.g. RESIDE training sets).
    Raises if a GT ID is duplicated or if a hazy image has no GT.
    """
    gts: dict[str, Path] = {}
    for p in sorted(Path(gt_dir).iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            key = image_id(p)
            if key in gts:
                raise ValueError(f"duplicate ground-truth id {key!r}: {gts[key]} and {p}")
            gts[key] = p
    pairs, missing = [], []
    for p in sorted(Path(hazy_dir).iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        gt = gts.get(image_id(p))
        (pairs.append((p, gt)) if gt else missing.append(p.name))
    if missing:
        raise ValueError(f"{len(missing)} hazy image(s) without ground truth, e.g. {missing[:3]}")
    if not pairs:
        raise ValueError(f"no image pairs found in {hazy_dir} / {gt_dir}")
    return pairs


def center_crop_to(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Center-crop ``img`` to (h, w). Only allowed to shrink."""
    h, w = shape
    H, W = img.shape[:2]
    if h > H or w > W:
        raise ValueError(f"cannot crop {img.shape[:2]} to larger {shape}")
    y, x = (H - h) // 2, (W - w) // 2
    return img[y:y + h, x:x + w]
