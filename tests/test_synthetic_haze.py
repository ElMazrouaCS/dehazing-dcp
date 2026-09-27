"""Controlled experiment: haze a scene with KNOWN A and t, check we recover them.

This validates the implementation against the image formation model itself,
without depending on any external dataset or reference image.
"""

import numpy as np

from dcp import DehazeParams, dehaze
from dcp.metrics import psnr
from dcp.prior import atmospheric_light, dark_channel, raw_transmission
from dcp.synthetic import ground_region

PATCH = 15


def test_atmospheric_light_is_recovered(synthetic_scene):
    I = synthetic_scene["I"]
    A_hat = atmospheric_light(I, dark_channel(I, PATCH))
    # In the sky t = 0.02, so I = 0.98 A + 0.02 J: error bounded by 0.02 * 230 ~ 5.
    np.testing.assert_allclose(A_hat, synthetic_scene["A"], atol=6)


def test_raw_transmission_matches_ground_truth(synthetic_scene):
    I, t = synthetic_scene["I"], synthetic_scene["t"]
    A_hat = atmospheric_light(I, dark_channel(I, PATCH))
    t_raw = raw_transmission(I, A_hat, PATCH, omega=1.0)
    err = np.abs(t_raw - t)[ground_region(synthetic_scene)]
    # The patch minimum returns the max of t over the patch: with a gradient of
    # 0.6 / 160 px, that bias is at most ~0.03.
    assert err.mean() < 0.03
    assert err.max() < 0.06


def test_omega_scales_the_haze_term_exactly(synthetic_scene):
    I = synthetic_scene["I"]
    A = synthetic_scene["A"]
    t1 = raw_transmission(I, A, PATCH, omega=1.0)
    t95 = raw_transmission(I, A, PATCH, omega=0.95)
    # t(omega) = 1 - omega * dark  =>  1 - t95 = 0.95 * (1 - t1) (before clipping)
    np.testing.assert_allclose(1 - t95, 0.95 * (1 - t1), atol=1e-6)


def test_guided_refinement_stays_close_to_ground_truth(synthetic_scene):
    r = dehaze(synthetic_scene["I"], DehazeParams(method="guided", omega=1.0))
    err = np.abs(r.t_refined - synthetic_scene["t"])[ground_region(synthetic_scene)]
    assert err.mean() < 0.05


def test_dehazing_recovers_the_clean_scene(synthetic_scene):
    region = ground_region(synthetic_scene)
    J_true = synthetic_scene["J"][region]
    hazy = synthetic_scene["I"][region]
    r = dehaze(synthetic_scene["I"], DehazeParams(method="guided", omega=1.0))
    baseline = psnr(hazy, J_true)
    recovered = psnr(r.J[region], J_true)
    assert recovered > baseline + 10, (baseline, recovered)
