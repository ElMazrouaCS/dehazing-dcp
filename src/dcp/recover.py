"""Scene radiance recovery and relative depth."""

from __future__ import annotations

import numpy as np


def recover_radiance(
    image: np.ndarray, transmission: np.ndarray, A: np.ndarray, t0: float = 0.1
) -> np.ndarray:
    """J(x) = (I(x) - A) / max(t(x), t0) + A, returned as uint8.

    ``t0`` bounds the amplification of noise where the transmission is tiny.
    """
    t = np.maximum(transmission, t0)[:, :, np.newaxis]
    J = (image.astype(np.float64) - A) / t + A
    return np.clip(J, 0.0, 255.0).astype(np.uint8)


def relative_depth(transmission: np.ndarray, t0: float = 0.1) -> np.ndarray:
    """Relative depth d(x) = -ln(t(x)), i.e. beta * true depth.

    The scattering coefficient beta is unknown, so this is only defined up to a
    scale factor: it orders pixels from near to far, it is not in metres.
    Values are bounded above by -ln(t0) because of the clamp.
    """
    return -np.log(np.maximum(transmission, t0))
