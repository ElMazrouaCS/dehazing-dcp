"""Full dehazing pipeline with explicit parameters and per-stage timings."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from dcp.prior import atmospheric_light, dark_channel, raw_transmission
from dcp.recover import recover_radiance, relative_depth
from dcp.refine import RefineInfo, refine_guided, refine_soft_matting

logger = logging.getLogger(__name__)

METHODS = ("guided", "soft_matting", "none")


@dataclass
class DehazeParams:
    """Every tunable of the pipeline. Defaults follow the paper where it gives one."""

    method: str = "guided"          # "guided", "soft_matting" or "none" (raw t~)
    patch_size: int = 15            # paper: 15
    omega: float = 0.95             # paper: 0.95
    t0: float = 0.1                 # paper: 0.1
    top_fraction: float = 0.001     # paper: brightest 0.1 % of the dark channel
    # guided filter (not in the 2009 paper; values chosen empirically)
    gf_radius: int = 40
    gf_eps: float = 1e-3
    # soft matting
    lambda_: float = 1e-4           # paper: 1e-4
    cg_tol: float = 1e-5
    cg_maxiter: int = 5000
    sm_max_side: int | None = 600   # memory workaround, see refine_soft_matting
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not 0.0 < self.omega <= 1.0:
            raise ValueError("omega must be in (0, 1]")
        if not 0.0 < self.t0 < 1.0:
            raise ValueError("t0 must be in (0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DehazeResult:
    J: np.ndarray                   # dehazed image, uint8 BGR
    dark: np.ndarray                # dark channel of the input, float32, 0-255
    A: np.ndarray                   # atmospheric light (B, G, R)
    t_raw: np.ndarray               # raw transmission, float64 in [0, 1]
    t_refined: np.ndarray           # refined transmission, float64 in [0, 1]
    depth: np.ndarray               # relative depth -ln(max(t, t0))
    params: DehazeParams
    refine_info: RefineInfo
    timings_s: dict[str, float] = field(default_factory=dict)


def dehaze(image: np.ndarray, params: DehazeParams | None = None) -> DehazeResult:
    """Run the dark-channel-prior pipeline on a BGR uint8 image."""
    p = params or DehazeParams()
    timings: dict[str, float] = {}

    def timed(name, fn, *args, **kwargs):
        start = time.perf_counter()
        out = fn(*args, **kwargs)
        timings[name] = time.perf_counter() - start
        return out

    total_start = time.perf_counter()
    dark = timed("dark_channel", dark_channel, image, p.patch_size)
    A = timed("atmospheric_light", atmospheric_light, image, dark, p.top_fraction)
    logger.info("atmospheric light A (B,G,R) = %s", np.round(A, 1))
    t_raw = timed("raw_transmission", raw_transmission, image, A, p.patch_size, p.omega)

    if p.method == "guided":
        t_ref, info = refine_guided(image, t_raw, p.gf_radius, p.gf_eps)
    elif p.method == "soft_matting":
        t_ref, info = refine_soft_matting(
            image, t_raw, p.lambda_, p.cg_tol, p.cg_maxiter, p.sm_max_side, p.allow_fallback
        )
    else:
        t_ref, info = t_raw, RefineInfo(requested_method="none", effective_method="none")
    timings["refine"] = info.timings_s.get("refine", 0.0)

    J = timed("recover", recover_radiance, image, t_ref, A, p.t0)
    depth = relative_depth(t_ref, p.t0)
    timings["total"] = time.perf_counter() - total_start

    return DehazeResult(J, dark, A, t_raw, t_ref, depth, p, info, timings)
