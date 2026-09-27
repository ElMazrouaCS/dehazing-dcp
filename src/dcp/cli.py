"""Command-line interface.

    dcp run IMAGE [options]        dehaze one image
    dcp compare IMAGE [options]    guided filter vs soft matting on one image
    dcp sweep IMAGE --param omega --values 0.75 0.85 0.95 1.0

Every parameter of the pipeline is exposed; the resolved parameters and what
actually happened (downscaling, solver convergence, timings) are written to a
JSON file next to the outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from dcp.metrics import evaluate_pair
from dcp.pipeline import METHODS, DehazeParams, DehazeResult, dehaze
from dcp.refine import SoftMattingError
from dcp.viz import colorize_depth, dark_to_u8, mosaic, pipeline_mosaic, unit_to_u8

logger = logging.getLogger("dcp")

SWEEPABLE = {"patch_size": int, "omega": float, "t0": float,
             "gf_radius": int, "gf_eps": float, "lambda_": float}


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def write(path: Path, img: np.ndarray) -> None:
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")
    logger.info("saved %s", path)


def report(result: DehazeResult) -> dict:
    return {
        "params": result.params.to_dict(),
        "atmospheric_light_bgr": [round(float(a), 2) for a in result.A],
        "refine": asdict(result.refine_info),
        "timings_s": {k: round(v, 4) for k, v in result.timings_s.items()},
    }


def params_from_args(args: argparse.Namespace, method: str | None = None) -> DehazeParams:
    return DehazeParams(
        method=method or args.method, patch_size=args.patch_size, omega=args.omega,
        t0=args.t0, top_fraction=args.top_fraction, gf_radius=args.gf_radius,
        gf_eps=args.gf_eps, lambda_=args.lambda_, cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        sm_max_side=None if args.sm_max_side <= 0 else args.sm_max_side,
        allow_fallback=not args.no_fallback,
    )


def metrics_with_baseline(image, outputs: dict[str, np.ndarray], gt_path) -> dict:
    """Metrics vs a reference, ALWAYS including the untouched input as baseline."""
    gt = load_image(gt_path)
    rows = {"identity (input)": evaluate_pair(image, gt)}
    rows.update({name: evaluate_pair(out, gt) for name, out in outputs.items()})
    for name, m in rows.items():
        logger.info("  %-18s PSNR %6.2f dB   SSIM %.4f", name, m["psnr"], m["ssim"])
    return rows


# --------------------------------------------------------------------------
# sub-commands
# --------------------------------------------------------------------------

def cmd_run(args) -> None:
    image = load_image(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    result = dehaze(image, params_from_args(args))
    info = result.refine_info
    logger.info("done in %.3f s (refinement: %s)", result.timings_s["total"],
                info.effective_method)

    write(out / f"{stem}_dehazed.png", result.J)
    write(out / f"{stem}_mosaic.png", pipeline_mosaic(image, result))
    if args.save_all:
        write(out / f"{stem}_dark_channel.png", dark_to_u8(result.dark))
        write(out / f"{stem}_transmission_raw.png", unit_to_u8(result.t_raw))
        write(out / f"{stem}_transmission_refined.png", unit_to_u8(result.t_refined))
        write(out / f"{stem}_depth.png", colorize_depth(result.depth, result.params.t0))
        np.save(out / f"{stem}_transmission_refined.npy", result.t_refined)

    rep = report(result)
    if args.gt:
        rep["metrics"] = metrics_with_baseline(image, {info.effective_method: result.J}, args.gt)
    (out / f"{stem}_report.json").write_text(json.dumps(rep, indent=2))


def cmd_compare(args) -> None:
    image = load_image(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    results = {m: dehaze(image, params_from_args(args, m)) for m in ("guided", "soft_matting")}
    for m, r in results.items():
        logger.info("%-13s total %.3f s, refinement %.3f s (effective: %s)", m,
                    r.timings_s["total"], r.timings_s["refine"],
                    r.refine_info.effective_method)
        write(out / f"{stem}_dehazed_{m}.png", r.J)

    gf, sm = results["guided"], results["soft_matting"]
    diff = np.abs(gf.t_refined - sm.t_refined)
    logger.info("|t_guided - t_soft_matting|: mean %.4f, max %.4f", diff.mean(), diff.max())

    fig = mosaic(
        [image, unit_to_u8(gf.t_refined), gf.J, unit_to_u8(sm.t_refined), sm.J,
         unit_to_u8(np.clip(diff * 5, 0, 1))],
        ["Input", "t guided", "Dehazed guided", f"t {sm.refine_info.effective_method}",
         "Dehazed soft matting", "|t_gf - t_sm| x5"],
    )
    write(out / f"{stem}_comparison.png", fig)

    rep = {m: report(r) for m, r in results.items()}
    rep["transmission_abs_diff"] = {"mean": float(diff.mean()), "max": float(diff.max())}
    if args.gt:
        rep["metrics"] = metrics_with_baseline(
            image, {m: r.J for m, r in results.items()}, args.gt)
    (out / f"{stem}_comparison.json").write_text(json.dumps(rep, indent=2))


def cmd_sweep(args) -> None:
    image = load_image(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cast = SWEEPABLE[args.param]

    panels, titles = [], []
    for raw in args.values:
        value = cast(raw)
        params = params_from_args(args)
        setattr(params, args.param, value)
        params.__post_init__()  # re-validate
        r = dehaze(image, params)
        panels.append(r.J)
        titles.append(f"{args.param}={value}")
    write(out / f"{stem}_sweep_{args.param}.png", mosaic(panels, titles))


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def add_pipeline_args(p: argparse.ArgumentParser) -> None:
    d = DehazeParams()
    g = p.add_argument_group("pipeline parameters")
    g.add_argument("--method", choices=METHODS, default=d.method)
    g.add_argument("--patch-size", type=int, default=d.patch_size)
    g.add_argument("--omega", type=float, default=d.omega)
    g.add_argument("--t0", type=float, default=d.t0)
    g.add_argument("--top-fraction", type=float, default=d.top_fraction,
                   help="fraction of brightest dark-channel pixels used for A")
    g.add_argument("--gf-radius", type=int, default=d.gf_radius)
    g.add_argument("--gf-eps", type=float, default=d.gf_eps)
    g.add_argument("--lambda", dest="lambda_", type=float, default=d.lambda_)
    g.add_argument("--cg-tol", type=float, default=d.cg_tol)
    g.add_argument("--cg-maxiter", type=int, default=d.cg_maxiter)
    g.add_argument("--sm-max-side", type=int, default=d.sm_max_side,
                   help="downscale soft matting above this side (<=0 disables)")
    g.add_argument("--no-fallback", action="store_true",
                   help="fail instead of returning the raw transmission if CG diverges")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dcp", description="Dark Channel Prior dehazing (He et al., CVPR 2009)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_ in [("run", "dehaze one image"),
                        ("compare", "guided filter vs soft matting"),
                        ("sweep", "vary one parameter")]:
        p = sub.add_parser(name, help=help_,
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("image")
        p.add_argument("--out", default="results")
        add_pipeline_args(p)
        if name in ("run", "compare"):
            p.add_argument("--gt", default=None,
                           help="reference image; metrics always include the input as baseline")
        if name == "run":
            p.add_argument("--save-all", action="store_true")
        if name == "sweep":
            p.add_argument("--param", choices=sorted(SWEEPABLE), required=True)
            p.add_argument("--values", nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        {"run": cmd_run, "compare": cmd_compare, "sweep": cmd_sweep}[args.command](args)
    except (FileNotFoundError, ValueError, OSError, SoftMattingError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
