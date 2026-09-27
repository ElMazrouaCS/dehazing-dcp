"""Synthetic hazy scenes with KNOWN ground truth (J, t, A), following the
image formation model I = J * t + A * (1 - t) (paper, eq. 1).

Used by the tests and by ``scripts/synthetic_experiment.py``.
"""

from __future__ import annotations

import numpy as np


def make_scene(
    seed: int = 0,
    h: int = 120,
    w: int = 160,
    block: int = 8,
    sky_rows: int = 16,
    t_near: float = 0.9,
    t_far: float = 0.3,
    t_sky: float = 0.02,
    A: tuple[float, float, float] = (235.0, 240.0, 250.0),
    layout: str = "gradient",
    n_objects: int = 6,
    t_object: float = 0.85,
) -> dict:
    """Build a scene where the dark channel prior holds by construction.

    - J: ``block``x``block`` tiles of random colours, one channel of every tile
      in [0, 5], so the dark channel of J is ~0 everywhere.
    - t: a haze-opaque "sky" band (``t_sky``) on top, then a horizontal gradient
      from ``t_near`` to ``t_far``.
    """
    rng = np.random.default_rng(seed)
    gh, gw = h // block, w // block
    colours = rng.uniform(60, 230, size=(gh, gw, 3))
    dark_idx = rng.integers(0, 3, size=(gh, gw))
    dark_vals = rng.uniform(0, 5, size=(gh, gw))
    np.put_along_axis(colours, dark_idx[..., None], dark_vals[..., None], axis=2)
    J = np.kron(colours, np.ones((block, block, 1)))

    t = np.tile(np.linspace(t_near, t_far, w), (h, 1))
    t[:sky_rows] = t_sky
    if layout == "objects":
        for _ in range(n_objects):
            # Block-aligned rectangles so depth edges coincide with colour edges.
            bh, bw = rng.integers(2, 5), rng.integers(2, 6)
            by = rng.integers(sky_rows // block + 1, gh - bh + 1)
            bx = rng.integers(0, gw - bw + 1)
            t[by * block:(by + bh) * block, bx * block:(bx + bw) * block] = t_object
    A_arr = np.asarray(A, dtype=np.float64)
    I = J * t[..., None] + A_arr * (1.0 - t[..., None])

    return {
        "I": np.clip(I + 0.5, 0, 255).astype(np.uint8),
        "J": J.astype(np.uint8),
        "t": t,
        "A": A_arr,
        "sky": sky_rows,
    }


def ground_region(scene: dict, patch_size: int = 15) -> tuple[slice, slice]:
    """Pixels away from the sky band and the borders by more than a patch radius."""
    m = patch_size // 2 + 1
    return (slice(scene["sky"] + m, -m), slice(m, -m))
