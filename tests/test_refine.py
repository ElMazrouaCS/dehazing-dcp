import numpy as np
import pytest

from dcp.refine import SoftMattingError, matting_laplacian, refine_guided, refine_soft_matting


@pytest.fixture
def small_image(rng):
    return rng.integers(0, 256, size=(30, 40, 3)).astype(np.uint8)


def test_guided_filter_keeps_a_constant_map_constant(small_image):
    t, info = refine_guided(small_image, np.full(small_image.shape[:2], 0.6), radius=5)
    np.testing.assert_allclose(t, 0.6, atol=1e-6)
    assert info.effective_method == "guided"


def test_guided_filter_output_in_unit_interval(small_image, rng):
    t, _ = refine_guided(small_image, rng.uniform(0, 1, small_image.shape[:2]), radius=5)
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_matting_laplacian_is_symmetric_with_zero_row_sums(rng):
    img = rng.uniform(0, 1, size=(12, 15, 3))
    L = matting_laplacian(img)
    assert abs(L - L.T).max() < 1e-8
    # Constant alpha has zero matting cost: L @ 1 = 0.
    np.testing.assert_allclose(L @ np.ones(L.shape[0]), 0.0, atol=1e-8)


def test_soft_matting_converges_and_reports_it(small_image, rng):
    t_raw = rng.uniform(0.2, 0.9, small_image.shape[:2])
    t, info = refine_soft_matting(small_image, t_raw)
    assert info.cg_converged is True
    assert info.effective_method == "soft_matting"
    assert info.downscaled is False
    assert t.shape == t_raw.shape


def test_downscaling_is_reported(small_image):
    t_raw = np.full(small_image.shape[:2], 0.5)
    t, info = refine_soft_matting(small_image, t_raw, max_side=20)
    assert info.downscaled is True
    assert info.working_size == (20, 15)
    assert t.shape == t_raw.shape


def test_non_convergence_is_flagged_not_silent(small_image, rng):
    t_raw = rng.uniform(0.2, 0.9, small_image.shape[:2])
    t, info = refine_soft_matting(small_image, t_raw, maxiter=1)
    assert info.cg_converged is False
    assert info.effective_method == "raw_fallback"
    np.testing.assert_array_equal(t, t_raw)


def test_non_convergence_can_be_made_fatal(small_image, rng):
    t_raw = rng.uniform(0.2, 0.9, small_image.shape[:2])
    with pytest.raises(SoftMattingError):
        refine_soft_matting(small_image, t_raw, maxiter=1, allow_fallback=False)
