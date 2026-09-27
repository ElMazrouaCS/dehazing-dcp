"""Reproduction of "Single Image Haze Removal Using Dark Channel Prior"
(He, Sun & Tang, CVPR 2009)."""

from dcp.pipeline import DehazeParams, DehazeResult, dehaze

__all__ = ["DehazeParams", "DehazeResult", "dehaze"]
__version__ = "1.0.0"
