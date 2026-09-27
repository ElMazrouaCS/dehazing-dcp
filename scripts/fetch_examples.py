"""Download the example images from Wikimedia Commons and write SOURCES.md.

The list of images lives in examples/sources.json (file name, Commons title,
role). Author and licence are NOT typed by hand: they are read from the
Commons API (extmetadata) at download time, so SOURCES.md always matches the
file page. Images are resized so that their longest side is 600 px (the
resize is recorded in SOURCES.md, as the licences require).

    python scripts/fetch_examples.py            # download missing images
    python scripts/fetch_examples.py --force    # re-download everything

To swap an image: edit its "title" in examples/sources.json and rerun with --force.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks for a descriptive User-Agent.
USER_AGENT = "dcp-dehazing-examples/1.0 (educational research reproduction; python-urllib)"
ALLOWED_LICENCE_PREFIXES = ("CC BY", "CC0", "Public domain", "PD")
MAX_SIDE = 600


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def image_info(title: str) -> dict:
    query = urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "imageinfo", "titles": title,
        "iiprop": "url|extmetadata", "iiurlwidth": "1200",
    })
    data = json.loads(http_get(f"{API}?{query}"))
    page = data["query"]["pages"][0]
    if page.get("missing") or "imageinfo" not in page:
        raise LookupError(f"{title!r} not found on Commons: fix examples/sources.json")
    info = page["imageinfo"][0]
    meta = info.get("extmetadata", {})

    def field(name: str) -> str:
        return strip_html(meta.get(name, {}).get("value", ""))

    return {
        "download_url": info.get("thumburl") or info["url"],
        "page_url": info["descriptionurl"],
        "author": field("Artist") or field("Credit") or "unknown",
        "licence": field("LicenseShortName") or "unknown",
        "licence_url": field("LicenseUrl"),
    }


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download existing images")
    args = ap.parse_args()

    entries = json.loads((EXAMPLES / "sources.json").read_text(encoding="utf-8"))
    rows, failed = [], 0
    for e in entries:
        target = EXAMPLES / e["file"]
        try:
            info = image_info(e["title"])
            if not info["licence"].startswith(ALLOWED_LICENCE_PREFIXES):
                raise ValueError(f"licence {info['licence']!r} is not CC BY/CC0/public "
                                 "domain: choose another image")
            if args.force or not target.exists():
                buf = np.frombuffer(http_get(info["download_url"]), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("downloaded file is not a readable image")
                img = resize_max_side(img, MAX_SIDE)
                cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            h, w = cv2.imread(str(target)).shape[:2]
            print(f"ok   {e['file']:<26} {w}x{h}  {info['licence']:<14} {info['author']}")
            rows.append((e, info, w, h))
        except Exception as exc:  # report every failure, keep going
            failed += 1
            print(f"FAIL {e['file']:<26} {exc}", file=sys.stderr)

    lines = [
        "# Example images",
        "",
        "Downloaded from Wikimedia Commons by `scripts/fetch_examples.py`; author and",
        "licence are read from the Commons API. All images were **resized** so that",
        f"the longest side is {MAX_SIDE} px and re-encoded as JPEG; no other change.",
        "",
        "| File | Use | Source | Author | Licence |",
        "|---|---|---|---|---|",
    ]
    for e, info, w, h in rows:
        lic = info["licence"]
        if info["licence_url"]:
            lic = f"[{lic}]({info['licence_url']})"
        name = e["title"].removeprefix("File:")
        author = info["author"].replace("|", "/")
        lines.append(f"| `{e['file']}` ({w}×{h}) | {e['role']} | "
                     f"[{name}]({info['page_url']}) | {author} | {lic} |")
    lines += [
        "",
        "None of these images has a haze-free reference: they are used for",
        "qualitative illustration only. Quantitative results come from",
        "`scripts/synthetic_experiment.py` (known ground truth) and",
        "`scripts/benchmark_dataset.py` (paired datasets).",
        "",
    ]
    (EXAMPLES / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {EXAMPLES / 'SOURCES.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
