"""Visualisation helpers. Kept out of the algorithm on purpose.

Maps are converted with FIXED scales (transmission 0-1 -> 0-255, dark channel
0-255 as is), never min-max stretched, so images of different methods or inputs
remain comparable.
"""

from __future__ import annotations

import cv2
import numpy as np


def unit_to_u8(x: np.ndarray) -> np.ndarray:
    """Map a [0, 1] array (e.g. a transmission) to uint8 with a fixed scale."""
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def dark_to_u8(dark: np.ndarray) -> np.ndarray:
    return np.clip(dark, 0, 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray, t0: float = 0.1) -> np.ndarray:
    """Colour-map a relative depth with a FIXED range [0, -ln(t0)].

    Blue = near, red = far (JET). Same colour = same relative depth, across
    images and methods.
    """
    d_max = -np.log(t0)
    d_u8 = (np.clip(depth / d_max, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def mosaic(
    panels: list[np.ndarray],
    titles: list[str],
    panel_height: int = 280,
    max_width: int = 1500,
) -> np.ndarray:
    """Titled grid of panels, all resized to the same height, wrapped by width."""
    titled = []
    for img, title in zip(panels, titles, strict=True):
        img = to_bgr(img)
        h, w = img.shape[:2]
        new_w = max(1, int(w * panel_height / h))
        img = cv2.resize(img, (new_w, panel_height), interpolation=cv2.INTER_AREA)
        bar = np.full((28, new_w, 3), 30, dtype=np.uint8)
        cv2.putText(bar, title, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
        titled.append(np.vstack([bar, img]))

    rows, row, row_w = [], [], 0
    for t in titled:
        if row and row_w + t.shape[1] > max_width:
            rows.append(np.hstack(row))
            row, row_w = [], 0
        row.append(t)
        row_w += t.shape[1]
    if row:
        rows.append(np.hstack(row))

    width = max(r.shape[1] for r in rows)
    rows = [
        np.hstack([r, np.full((r.shape[0], width - r.shape[1], 3), 30, np.uint8)])
        if r.shape[1] < width else r
        for r in rows
    ]
    return np.vstack(rows)


def pipeline_mosaic(image: np.ndarray, result) -> np.ndarray:
    """Six-panel figure: input, dark channel, raw t, refined t, output, depth."""
    method = result.refine_info.effective_method
    return mosaic(
        [image, dark_to_u8(result.dark), unit_to_u8(result.t_raw),
         unit_to_u8(result.t_refined), result.J,
         colorize_depth(result.depth, result.params.t0)],
        ["1. Input", "2. Dark channel", "3. Transmission (raw)",
         f"4. Transmission ({method})", "5. Dehazed", "6. Relative depth"],
    )
