"""Timing benchmark: median and IQR over repeated runs, per pipeline stage.

    python scripts/benchmark_timing.py examples/forbidden_city_smog.jpg --repeats 5
    python scripts/benchmark_timing.py examples/forbidden_city_smog.jpg --size-sweep

The environment (CPU, library versions) is stored with the results. On
Windows the CPU name reported by Python is vague: pass --cpu-label.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from _env import environment

from dcp import DehazeParams, dehaze


def summarize(values: list[float]) -> dict:
    q = np.percentile(values, [25, 50, 75])
    return {"median": float(q[1]), "q1": float(q[0]), "q3": float(q[2]),
            "min": min(values), "n": len(values)}


def time_method(image, method: str, repeats: int, warmup: int = 1) -> dict:
    for _ in range(warmup):
        dehaze(image, DehazeParams(method=method))
    per_stage: dict[str, list[float]] = {}
    info = None
    for _ in range(repeats):
        r = dehaze(image, DehazeParams(method=method))
        info = r.refine_info
        stages = dict(r.timings_s)
        stages.update({f"refine.{k}": v for k, v in info.timings_s.items() if k != "refine"})
        for k, v in stages.items():
            per_stage.setdefault(k, []).append(v)
    return {
        "effective_method": info.effective_method,
        "downscaled": info.downscaled,
        "working_size": info.working_size,
        "cg_iterations": info.cg_iterations,
        "stages": {k: summarize(v) for k, v in per_stage.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["guided", "soft_matting"])
    ap.add_argument("--size-sweep", action="store_true",
                    help="also time each method at several image widths (aspect kept)")
    ap.add_argument("--widths", nargs="+", type=int, default=[150, 300, 450, 600])
    ap.add_argument("--cpu-label", default=None)
    ap.add_argument("--out", default="results/timing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    h, w = image.shape[:2]
    result = {"image": args.image, "size": [w, h], "repeats": args.repeats,
              "env": environment(args.cpu_label), "methods": {}}

    for m in args.methods:
        res = time_method(image, m, args.repeats)
        result["methods"][m] = res
        tot = res["stages"]["total"]
        print(f"{m:<13} total median {tot['median']:.3f} s "
              f"[IQR {tot['q1']:.3f}-{tot['q3']:.3f}], refine median "
              f"{res['stages']['refine']['median']:.3f} s, effective={res['effective_method']}")

    if args.size_sweep:
        result["size_sweep"] = []
        for tw in args.widths:
            th = round(h * tw / w)
            small = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
            for m in args.methods:
                res = time_method(small, m, max(3, args.repeats // 2))
                med = res["stages"]["refine"]["median"]
                result["size_sweep"].append({"width": tw, "height": th, "pixels": tw * th,
                                             "method": m, "refine_median_s": med})
                print(f"  {tw}x{th:<5} {m:<13} refine {med:.4f} s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.image).stem}_timing.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
