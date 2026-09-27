import json

import cv2
import numpy as np
import pytest

from dcp import DehazeParams, dehaze
from dcp.cli import main


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        DehazeParams(method="magic")
    with pytest.raises(ValueError):
        DehazeParams(omega=1.5)


def test_method_none_returns_raw_transmission(synthetic_scene):
    r = dehaze(synthetic_scene["I"], DehazeParams(method="none"))
    np.testing.assert_array_equal(r.t_refined, r.t_raw)
    assert r.J.dtype == np.uint8 and r.J.shape == synthetic_scene["I"].shape
    assert set(r.timings_s) >= {"dark_channel", "atmospheric_light", "refine", "total"}


def test_cli_run_writes_outputs_and_report(tmp_path, synthetic_scene):
    img_path = tmp_path / "scene.png"
    cv2.imwrite(str(img_path), synthetic_scene["I"])
    gt_path = tmp_path / "gt.png"
    cv2.imwrite(str(gt_path), synthetic_scene["J"])

    code = main(["run", str(img_path), "--out", str(tmp_path / "out"),
                 "--save-all", "--gt", str(gt_path)])
    assert code == 0
    report = json.loads((tmp_path / "out" / "scene_report.json").read_text())
    assert report["refine"]["effective_method"] == "guided"
    # The untouched input is always reported as a baseline.
    assert "identity (input)" in report["metrics"]
    assert (tmp_path / "out" / "scene_transmission_refined.png").exists()


def test_cli_missing_file_returns_error_code(tmp_path):
    assert main(["run", str(tmp_path / "nope.png")]) == 1
