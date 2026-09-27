"""Transmission refinement: soft matting (paper) and guided filter (fast).

Both functions return ``(t_refined, RefineInfo)``. ``RefineInfo`` records what
actually happened (downscaling, solver convergence, fallback, timings) so that
no behaviour change is ever silent.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)


@dataclass
class RefineInfo:
    """What the refinement step actually did."""

    requested_method: str
    #: "guided", "soft_matting" or "raw_fallback" (soft matting failed and the
    #: raw transmission was returned instead).
    effective_method: str
    downscaled: bool = False
    working_size: tuple[int, int] | None = None  # (width, height) actually solved
    cg_converged: bool | None = None
    cg_iterations: int | None = None
    timings_s: dict[str, float] = field(default_factory=dict)


class SoftMattingError(RuntimeError):
    """Raised when the soft matting solver fails and fallback is disabled."""


# --------------------------------------------------------------------------
# Guided filter
# --------------------------------------------------------------------------

def refine_guided(
    image: np.ndarray, t_raw: np.ndarray, radius: int = 40, eps: float = 1e-3
) -> tuple[np.ndarray, RefineInfo]:
    """Guided filter (He et al., ECCV 2010) with the grey-level image as guide.

    In every window the output is an affine function of the guide,
    q = a * I + b, which transfers the image edges to the transmission map.
    Only box filters are involved, so the cost is O(N) whatever the radius.

    Note: ``radius`` is in pixels, so its effect depends on image resolution.
    """
    start = time.perf_counter()
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    p = t_raw.astype(np.float64)
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.blur(x, ksize)

    mean_I = box(I)
    mean_p = box(p)
    var_I = box(I * I) - mean_I * mean_I
    cov_Ip = box(I * p) - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    q = np.clip(box(a) * I + box(b), 0.0, 1.0)

    info = RefineInfo(
        requested_method="guided",
        effective_method="guided",
        working_size=(image.shape[1], image.shape[0]),
        timings_s={"refine": time.perf_counter() - start},
    )
    return q, info


# --------------------------------------------------------------------------
# Soft matting
# --------------------------------------------------------------------------

def matting_laplacian(
    img_norm: np.ndarray, eps: float = 1e-7, win_rad: int = 1
) -> sp.csr_matrix:
    """Matting Laplacian of Levin et al. (CVPR 2006), shape (H*W, H*W).

    Vectorised over the windows of one image row at a time.

    Memory: about (H-2r)(W-2r) * (2r+1)^4 COO entries (81 per window for r=1),
    i.e. ~350 MB for a 600x450 image before CSR conversion. This is why large
    images have to be downscaled (see ``refine_soft_matting``).
    """
    h, w, c = img_norm.shape
    ws = 2 * win_rad + 1
    k = ws * ws
    n_pix = h * w
    hv, wv = h - 2 * win_rad, w - 2 * win_rad
    if hv <= 0 or wv <= 0:
        raise ValueError("image too small for the requested window radius")

    total = hv * wv * k * k
    rows_coo = np.empty(total, dtype=np.int32)
    cols_coo = np.empty(total, dtype=np.int32)
    vals_coo = np.empty(total, dtype=np.float64)

    cols_win = np.arange(wv)[:, None] + np.arange(ws)[None, :]
    eye_c = np.eye(c) * (eps / k)
    eye_k = np.eye(k)[None, :, :]

    ptr = 0
    for vi in range(hv):
        i = vi + win_rad
        rows_win = np.arange(i - win_rad, i + win_rad + 1)

        # All windows centred on this row: (wv, k, c)
        wins = img_norm[rows_win[None, :, None], cols_win[:, None, :], :]
        wins = wins.reshape(wv, k, c)

        mu = wins.mean(axis=1)
        e_xxT = np.einsum("nkc,nkd->ncd", wins, wins) / k
        sigma = e_xxT - np.einsum("nc,nd->ncd", mu, mu) + eye_c
        sig_inv = np.linalg.inv(sigma)

        diff = wins - mu[:, None, :]
        tmp = np.einsum("nkc,ncd->nkd", diff, sig_inv)
        quad = np.einsum("npc,nqc->npq", tmp, diff)
        lap = eye_k - (1.0 / k) * (1.0 + quad)

        pix = (rows_win[None, :, None] * w + cols_win[:, None, :]).reshape(wv, k)
        n_row = wv * k * k
        rows_coo[ptr:ptr + n_row] = pix[:, :, None].repeat(k, axis=2).ravel()
        cols_coo[ptr:ptr + n_row] = pix[:, None, :].repeat(k, axis=1).ravel()
        vals_coo[ptr:ptr + n_row] = lap.ravel()
        ptr += n_row

    return sp.coo_matrix(
        (vals_coo[:ptr], (rows_coo[:ptr], cols_coo[:ptr])), shape=(n_pix, n_pix)
    ).tocsr()


def _conjugate_gradient(A, b, tol, maxiter, callback):
    # scipy >= 1.12 renamed ``tol`` to ``rtol``.
    try:
        return spla.cg(A, b, rtol=tol, maxiter=maxiter, callback=callback)
    except TypeError:
        return spla.cg(A, b, tol=tol, maxiter=maxiter, callback=callback)


def refine_soft_matting(
    image: np.ndarray,
    t_raw: np.ndarray,
    lambda_: float = 1e-4,
    tol: float = 1e-5,
    maxiter: int = 5000,
    max_side: int | None = 600,
    allow_fallback: bool = True,
) -> tuple[np.ndarray, RefineInfo]:
    """Soft matting refinement: solve (L + lambda I) t = lambda t~ (paper, eq. 15).

    Args:
        lambda_: data-term weight (paper: 1e-4).
        tol, maxiter: conjugate-gradient stopping criteria.
        max_side: if the longest side exceeds this, the system is solved on a
            downscaled copy and the result is upsampled. This is a memory
            workaround, NOT the paper's method; it is recorded in the returned
            info and logged as a warning. ``None`` disables it.
        allow_fallback: if the solver does not converge, return the raw
            transmission (flagged as ``effective_method="raw_fallback"``)
            instead of raising ``SoftMattingError``.
    """
    total_start = time.perf_counter()
    h, w = image.shape[:2]
    scale = 1.0 if max_side is None else min(max_side / max(h, w), 1.0)
    info = RefineInfo(requested_method="soft_matting", effective_method="soft_matting")

    if scale < 1.0:
        sw, sh = int(w * scale), int(h * scale)
        logger.warning(
            "soft matting: %dx%d exceeds max_side=%d, solving at %dx%d then "
            "upsampling (approximation of the paper's method)",
            w, h, max_side, sw, sh,
        )
        img_small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        t_small_raw = cv2.resize(
            t_raw.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA
        ).astype(np.float64)
        info.downscaled = True
    else:
        sw, sh = w, h
        img_small, t_small_raw = image, t_raw
    info.working_size = (sw, sh)

    start = time.perf_counter()
    L = matting_laplacian(img_small.astype(np.float64) / 255.0)
    info.timings_s["laplacian"] = time.perf_counter() - start

    n_pix = sh * sw
    system = L + lambda_ * sp.eye(n_pix, format="csr")
    rhs = lambda_ * t_small_raw.ravel()

    iterations = 0

    def count(_xk):
        nonlocal iterations
        iterations += 1

    start = time.perf_counter()
    t_flat, status = _conjugate_gradient(system, rhs, tol, maxiter, count)
    info.timings_s["solve"] = time.perf_counter() - start
    info.cg_iterations = iterations
    info.cg_converged = status == 0

    if status != 0:
        msg = f"conjugate gradient did not converge (status={status}, {iterations} iterations)"
        if not allow_fallback:
            raise SoftMattingError(msg)
        logger.warning("%s: returning the RAW transmission, flagged raw_fallback", msg)
        info.effective_method = "raw_fallback"
        t_small = t_small_raw
    else:
        t_small = np.clip(t_flat.reshape(sh, sw), 0.0, 1.0)

    if info.downscaled:
        t_small = cv2.resize(
            t_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float64)
    if info.effective_method == "raw_fallback":
        t_small = t_raw.astype(np.float64)  # full-resolution raw map, not a resampled one

    info.timings_s["refine"] = time.perf_counter() - total_start
    return np.clip(t_small, 0.0, 1.0), info
