import numpy as np
import pytest

from dcp.prior import atmospheric_light, dark_channel, raw_transmission


def brute_force_dark_channel(img, patch):
    """Reference implementation: explicit loops, border windows clipped."""
    h, w, _ = img.shape
    r = patch // 2
    mins = img.min(axis=2)
    out = np.empty((h, w))
    for y in range(h):
        for x in range(w):
            out[y, x] = mins[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].min()
    return out


@pytest.mark.parametrize("patch", [1, 3, 7, 15])
def test_dark_channel_matches_brute_force(rng, patch):
    img = rng.integers(0, 256, size=(23, 31, 3)).astype(np.uint8)
    np.testing.assert_array_equal(dark_channel(img, patch), brute_force_dark_channel(img, patch))


def test_dark_channel_of_constant_image_is_its_min_channel():
    img = np.full((20, 20, 3), (40, 90, 200), dtype=np.uint8)
    np.testing.assert_array_equal(dark_channel(img, 7), 40)


def test_dark_channel_rejects_grey_images():
    with pytest.raises(ValueError):
        dark_channel(np.zeros((10, 10), np.uint8))


def test_atmospheric_light_ignores_a_bright_object_with_a_dark_pixel(rng):
    """The reason for using the dark channel: a small white object (one dark
    pixel inside it) must not be picked over a large, uniformly hazy region."""
    img = rng.integers(0, 100, size=(60, 60, 3)).astype(np.uint8)
    img[:20, :] = (180, 190, 200)          # hazy sky
    img[40:50, 40:50] = 255                # white car ...
    img[45, 45] = 0                        # ... with a dark detail
    A = atmospheric_light(img, dark_channel(img, 15))
    np.testing.assert_array_equal(A, (180, 190, 200))


def test_raw_transmission_is_in_unit_interval(rng):
    img = rng.integers(0, 256, size=(30, 30, 3)).astype(np.uint8)
    t = raw_transmission(img, np.array([200.0, 200.0, 200.0]))
    assert t.min() >= 0.0 and t.max() <= 1.0
