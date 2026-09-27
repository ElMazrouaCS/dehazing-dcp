import cv2
import numpy as np
import pytest

from dcp.datasets import center_crop_to, pair_images


def touch_img(path):
    cv2.imwrite(str(path), np.zeros((4, 4, 3), np.uint8))


def test_pairs_ohaze_and_sots_naming(tmp_path):
    hazy, gt = tmp_path / "hazy", tmp_path / "gt"
    hazy.mkdir()
    gt.mkdir()
    for name in ["01_outdoor_hazy.jpg", "0002_0.8_0.2.jpg"]:
        touch_img(hazy / name)
    for name in ["01_outdoor_GT.jpg", "0002.png"]:
        touch_img(gt / name)
    pairs = pair_images(hazy, gt)
    assert [(h.name, g.name) for h, g in pairs] == [
        ("0002_0.8_0.2.jpg", "0002.png"), ("01_outdoor_hazy.jpg", "01_outdoor_GT.jpg")]


def test_missing_ground_truth_is_an_error(tmp_path):
    hazy, gt = tmp_path / "hazy", tmp_path / "gt"
    hazy.mkdir()
    gt.mkdir()
    touch_img(hazy / "07_hazy.png")
    with pytest.raises(ValueError):
        pair_images(hazy, gt)


def test_center_crop():
    img = np.arange(6 * 8).reshape(6, 8)
    assert center_crop_to(img, (4, 4)).tolist() == img[1:5, 2:6].tolist()
    with pytest.raises(ValueError):
        center_crop_to(img, (7, 8))
