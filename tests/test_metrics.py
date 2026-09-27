import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dcp.metrics import psnr, ssim


@pytest.fixture
def pair(rng):
    a = rng.integers(0, 256, size=(64, 80, 3)).astype(np.uint8)
    noise = rng.normal(0, 15, size=a.shape)
    b = np.clip(a + noise, 0, 255).astype(np.uint8)
    return a, b


def test_psnr_identical_is_infinite(pair):
    assert psnr(pair[0], pair[0]) == float("inf")


def test_psnr_matches_skimage(pair):
    a, b = pair
    assert psnr(a, b) == pytest.approx(peak_signal_noise_ratio(a, b, data_range=255), abs=1e-9)


def test_ssim_matches_skimage(pair):
    a, b = pair
    ref = structural_similarity(a, b, data_range=255, channel_axis=2,
                                gaussian_weights=True, sigma=1.5,
                                use_sample_covariance=False)
    assert ssim(a, b) == pytest.approx(ref, abs=1e-3)


def test_ssim_identical_is_one(pair):
    assert ssim(pair[0], pair[0]) == pytest.approx(1.0)


def test_shape_mismatch_raises_instead_of_resizing(pair):
    with pytest.raises(ValueError):
        psnr(pair[0], pair[0][:-1])
