"""Environment description stored next to every benchmark result."""

from __future__ import annotations

import os
import platform
import sys


def cpu_name() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def environment(cpu_label: str | None = None) -> dict:
    import cv2
    import numpy
    import scipy

    return {
        "cpu": cpu_label or cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
    }
