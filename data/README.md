# Datasets (not committed)

The dataset benchmark (`scripts/benchmark_dataset.py`) expects a folder of
hazy images and a folder of clean references. Files are paired by the ID before
the first underscore (`dcp/datasets.py`).

| Dataset | Content | Naming |
|---|---|---|
| RESIDE **SOTS-outdoor** | synthetic haze on real outdoor photos, 500 pairs | `0001_0.8_0.2.jpg` ↔ `0001.png` |
| **O-HAZE** (NTIRE 2018) | real haze produced by haze machines, 45 outdoor pairs, high resolution | `01_outdoor_hazy.jpg` ↔ `01_outdoor_GT.jpg` |

Download them from their official pages, check the licence/terms of use, and
place them as:

```
data/sots_outdoor/hazy/   data/sots_outdoor/clear/
data/ohaze/hazy/          data/ohaze/GT/
```

Notes:
- O-HAZE images are several megapixels: soft matting would be solved on a
  downscaled copy. Use `--max-side 600` so both methods run at the same
  resolution (the resize is recorded in `summary.json`).
- If a reference and its hazy image differ in size, the script stops; pass
  `--size-mismatch crop` to center-crop the reference (the count is recorded).
