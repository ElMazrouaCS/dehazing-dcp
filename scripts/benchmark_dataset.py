"""Full-reference benchmark on a paired dehazing dataset.

Always reports the untouched hazy input ("identity") as a baseline, and the
per-image paired difference of every method against it.

    python scripts/benchmark_dataset.py --hazy data/sots_outdoor/hazy \\
        --gt data/sots_outdoor/clear --limit 30 --seed 0 --out results/sots_outdoor

Outputs: per_image.csv, summary.json (mean ± std, paired differences, what the
refinement actually did, environment).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
from collections import Counter
from pathlib import Path

import cv2
from _env import environment

from dcp import DehazeParams, dehaze
from dcp.datasets import center_crop_to, pair_images
from dcp.metrics import evaluate_pair


def resize_max_side(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def mean_std(vals):
    return {"mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hazy", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--methods", nargs="+", default=["guided", "soft_matting"])
    ap.add_argument("--limit", type=int, default=None, help="random subset size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=None,
                    help="resize hazy AND gt so the longest side is at most this (recorded)")
    ap.add_argument("--size-mismatch", choices=["error", "crop"], default="error",
                    help="if gt and hazy sizes differ: fail, or center-crop the gt (counted)")
    ap.add_argument("--cpu-label", default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    pairs = pair_images(args.hazy, args.gt)
    n_available = len(pairs)
    if args.limit and args.limit < len(pairs):
        pairs = sorted(random.Random(args.seed).sample(pairs, args.limit))

    rows, cropped, effective = [], 0, Counter()
    for i, (hp, gp) in enumerate(pairs, 1):
        hazy, gt = cv2.imread(str(hp)), cv2.imread(str(gp))
        if args.max_side:
            hazy, gt = resize_max_side(hazy, args.max_side), resize_max_side(gt, args.max_side)
        if hazy.shape != gt.shape:
            if args.size_mismatch == "error":
                raise SystemExit(f"size mismatch {hp.name} {hazy.shape} vs {gp.name} {gt.shape}; "
                                 "use --size-mismatch crop to center-crop the ground truth")
            gt = center_crop_to(gt, hazy.shape[:2])
            cropped += 1

        outputs = {"identity": (hazy, None)}
        for m in args.methods:
            r = dehaze(hazy, DehazeParams(method=m))
            outputs[m] = (r.J, r)
            effective[f"{m}->{r.refine_info.effective_method}"] += 1
        for m, (img, r) in outputs.items():
            row = {"image": hp.name, "method": m, **evaluate_pair(img, gt)}
            row["time_s"] = r.timings_s["total"] if r else 0.0
            rows.append(row)
        print(f"[{i}/{len(pairs)}] {hp.name}")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    methods = ["identity", *args.methods]
    by = {(r["image"], r["method"]): r for r in rows}
    images = sorted({r["image"] for r in rows})
    summary = {m: {k: mean_std([by[(im, m)][k] for im in images]) for k in ("psnr", "ssim")}
               for m in methods}
    paired = {}
    for m in args.methods:
        for k in ("psnr", "ssim"):
            d = [by[(im, m)][k] - by[(im, "identity")][k] for im in images]
            paired[f"{m} - identity ({k})"] = {**mean_std(d), "wins": sum(x > 0 for x in d)}
    if {"guided", "soft_matting"} <= set(args.methods):
        for k in ("psnr", "ssim"):
            d = [by[(im, "soft_matting")][k] - by[(im, "guided")][k] for im in images]
            paired[f"soft_matting - guided ({k})"] = {**mean_std(d), "wins": sum(x > 0 for x in d)}

    report = {
        "hazy_dir": str(args.hazy), "gt_dir": str(args.gt),
        "n_available": n_available, "n_evaluated": len(images),
        "subset": {"limit": args.limit, "seed": args.seed},
        "max_side": args.max_side, "gt_center_cropped": cropped,
        "refinement_effective": dict(effective),
        "summary": summary, "paired": paired, "env": environment(args.cpu_label),
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2))

    print(f"\n{len(images)} images (of {n_available}), gt cropped: {cropped}")
    for m in methods:
        s = summary[m]
        print(f"{m:<13} PSNR {s['psnr']['mean']:6.2f} ± {s['psnr']['std']:.2f}   "
              f"SSIM {s['ssim']['mean']:.3f} ± {s['ssim']['std']:.3f}")
    for k, v in paired.items():
        print(f"  {k:<32} {v['mean']:+.3f} ± {v['std']:.3f}   better on {v['wins']}/{v['n']}")
    print(f"refinement effective: {dict(effective)}")


if __name__ == "__main__":
    main()
