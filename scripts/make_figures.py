"""Regenerate every figure of the README from the example images.

    python scripts/make_figures.py --out docs/figures

All maps use fixed scales (see dcp.viz): transmission 0 = black, 1 = white;
depth colour map spans [0, -ln(t0)].
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from dcp import DehazeParams, dehaze
from dcp.synthetic import make_scene
from dcp.viz import colorize_depth, mosaic, pipeline_mosaic, unit_to_u8

EX = Path(__file__).resolve().parent.parent / "examples"


def load(name: str) -> np.ndarray:
    img = cv2.imread(str(EX / name))
    if img is None:
        raise SystemExit(f"missing examples/{name}: run python scripts/fetch_examples.py first")
    return img


def save(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)

    # full pipeline on one image.
    main_img = load("forbidden_city_smog.jpg")
    save(out / "pipeline.jpg", pipeline_mosaic(main_img, dehaze(main_img)))

    # Before / after on the example images (guided filter, paper defaults).
    panels, titles = [], []
    # Dense urban smog (forbidden_city_smog.jpg) stays out of this figure: it
    # is a poor "it works" demo (noise, colour cast) even though it is fine
    # for the technical pipeline figure above. These two have real depth
    # structure and a roughly neutral atmospheric light, which is the case
    # the paper's model fits best.
    demo_names = ["aerial_perspective_hills.jpg", "morning_mist_forest.jpg"]
    for name in demo_names:
        img = load(name)
        panels += [img, dehaze(img).J]
        titles += ["Input", "Dehazed"]
    save(out / "before_after.jpg", mosaic(panels, titles, max_width=1300))

    # Failure cases, run with the paper's DEFAULT parameters (no tuning).
    for name in ["snow_fog.jpg", "sunset_sky.jpg"]:
        img = load(name)
        r = dehaze(img)
        fig = mosaic(
            [img, r.J, unit_to_u8(r.t_refined), colorize_depth(r.depth, r.params.t0)],
            ["Input", "Dehazed (default params)", "Transmission (white = 1)",
             "Relative depth (blue = near)"], max_width=2000)
        save(out / f"failure_{Path(name).stem}.jpg", fig)

    # Synthetic scene with known ground truth
    sc = make_scene(seed=0, layout="objects")
    rs = {m: dehaze(sc["I"], DehazeParams(method=m, omega=1.0))
          for m in ("none", "guided", "soft_matting")}
    fig = mosaic(
        [sc["J"], sc["I"], unit_to_u8(sc["t"]), unit_to_u8(rs["none"].t_refined),
         unit_to_u8(rs["guided"].t_refined), unit_to_u8(rs["soft_matting"].t_refined)],
        ["Clean scene J (known)", "Hazed I", "True t", "Raw t~",
         "t guided filter", "t soft matting"], panel_height=240, max_width=1200)
    save(out / "synthetic_transmission.jpg", fig)


if __name__ == "__main__":
    main()
