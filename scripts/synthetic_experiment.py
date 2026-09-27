"""Controlled experiment on synthetic haze with known ground truth.

For several random scenes (see dcp.synthetic), compare:
  identity (hazy input), raw transmission, guided filter, soft matting
on transmission error (MAE vs true t) and PSNR/SSIM vs the true clean scene.
omega = 1 so that the target is the exact model (no haze kept on purpose).

    python scripts/synthetic_experiment.py --layout gradient
    python scripts/synthetic_experiment.py --layout objects

"gradient": smooth depth only (tests correctness of the prior/A/t estimation).
"objects": adds depth discontinuities aligned with colour edges (tests the
refinement, which exists to fix halos at such edges).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from _env import environment

from dcp import DehazeParams, dehaze
from dcp.metrics import psnr, ssim
from dcp.synthetic import ground_region, make_scene

METHODS = ["identity", "none", "guided", "soft_matting"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--layout", choices=["gradient", "objects"], default="objects")
    ap.add_argument("--out", default="results/synthetic")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.seeds):
        scene = make_scene(seed=seed, layout=args.layout)
        reg = ground_region(scene)
        J_true = scene["J"][reg]
        for m in METHODS:
            if m == "identity":
                J, t_mae = scene["I"], float("nan")
            else:
                r = dehaze(scene["I"], DehazeParams(method=m, omega=1.0))
                J = r.J
                t_mae = float(np.abs(r.t_refined - scene["t"])[reg].mean())
            rows.append({"seed": seed, "method": m, "t_mae": t_mae,
                         "psnr": psnr(J[reg], J_true), "ssim": ssim(J[reg], J_true)})

    out = out / args.layout
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_scene.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for m in METHODS:
        sel = [r for r in rows if r["method"] == m]
        summary[m] = {}
        for k in ("t_mae", "psnr", "ssim"):
            vals = [r[k] for r in sel if not np.isnan(r[k])]
            if vals:
                summary[m][k] = {"mean": statistics.fmean(vals), "std": statistics.stdev(vals)}
    # Paired differences (same scene): is the refinement better than raw t~?
    paired = {}
    by = {(r["seed"], r["method"]): r for r in rows}
    for m in ("guided", "soft_matting"):
        d = [by[(s, m)]["psnr"] - by[(s, "none")]["psnr"] for s in range(args.seeds)]
        paired[f"{m} - none (PSNR dB)"] = {
            "mean": statistics.fmean(d), "std": statistics.stdev(d),
            "wins": sum(x > 0 for x in d), "n": len(d)}
    json.dump({"n_scenes": args.seeds, "layout": args.layout, "summary": summary,
               "paired": paired, "env": environment()},
              open(out / "summary.json", "w"), indent=2)

    print(f"{'method':<14}{'t MAE':>16}{'PSNR (dB)':>18}{'SSIM':>18}")
    def fmt(stats: dict, key: str, prec: int) -> str:
        if key not in stats:
            return "—"
        return f"{stats[key]['mean']:.{prec}f} ± {stats[key]['std']:.{prec}f}"

    for m, st in summary.items():
        print(f"{m:<14}{fmt(st, 't_mae', 3):>16}{fmt(st, 'psnr', 2):>18}{fmt(st, 'ssim', 3):>18}")
    for k, v in paired.items():
        print(f"paired {k}: {v['mean']:+.2f} ± {v['std']:.2f}, better on {v['wins']}/{v['n']}")


if __name__ == "__main__":
    main()
