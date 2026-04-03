#!/usr/bin/env python3
import csv
import glob
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

# =======================
# SETTINGS
# =======================

INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Impact/2026-04-03/*_cal.csv"
INPUT_FILES = None

OUT_DIR = Path(r"G:/My Drive/Project SHM/Data/Impact/2026-04-03")

TIME_COLUMN = "recv_time_s"

NUM_SENSORS = 2
AXES = ("x", "y", "z")

# Kalibrerade kolumner från calibration.py
def cal_col(sensor_idx: int, axis: str) -> str:
    return f"s{sensor_idx}_a{axis}_g_cal"

# =======================
# POST-PROCESSING
# =======================

APPLY_HPF = True
HPF_CUTOFF_HZ = 0.35
HPF_ORDER = 2

APPLY_MEAN_REMOVAL = True

# Om båda är True:
# False = mean removal först, sedan HPF
# True  = HPF först, sedan mean removal
MEAN_REMOVAL_AFTER_HPF = True

OUTPUT_SUFFIX = "_post"

# =======================
# Helpers
# =======================

def resolve_files():
    if INPUT_FILES is not None:
        files = [Path(p) for p in INPUT_FILES]
    else:
        files = [Path(p) for p in sorted(glob.glob(INPUT_GLOB))]
    if not files:
        raise ValueError("Hittade inga CSV-filer.")
    return files

def safe_float(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    return float(s)

def format_float(v, decimals=9):
    if not np.isfinite(v):
        return ""
    return f"{v:.{decimals}f}"

def mean_remove(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    good = np.isfinite(y)
    if np.count_nonzero(good) > 0:
        y[good] = y[good] - np.mean(y[good])
    return y

def butter_hpf_zero_phase(x: np.ndarray, fs: float, cutoff_hz: float, order: int) -> np.ndarray:
    y = np.full_like(x, np.nan, dtype=np.float64)
    good = np.isfinite(x)
    min_len = max(10, 3 * (order + 1))

    if np.count_nonzero(good) < min_len:
        return y

    wn = cutoff_hz / (0.5 * fs)
    if not (0 < wn < 1):
        raise ValueError(f"Ogiltig HPF cutoff: cutoff={cutoff_hz} Hz, fs={fs} Hz")

    b, a = butter(order, wn, btype="highpass")

    if np.all(good):
        return filtfilt(b, a, x)

    idx = np.where(good)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    blocks = np.split(idx, splits + 1)

    for blk in blocks:
        if len(blk) < min_len:
            continue
        seg = x[blk]
        try:
            y[blk] = filtfilt(b, a, seg)
        except ValueError:
            pass

    return y

# =======================
# Main
# =======================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = resolve_files()

    print(f"Found {len(files)} files.")
    print(
        f"APPLY_HPF={APPLY_HPF}, "
        f"HPF_CUTOFF_HZ={HPF_CUTOFF_HZ}, "
        f"APPLY_MEAN_REMOVAL={APPLY_MEAN_REMOVAL}, "
        f"MEAN_REMOVAL_AFTER_HPF={MEAN_REMOVAL_AFTER_HPF}"
    )

    for fpath in files:
        with fpath.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            rows = list(r)

        if not rows:
            print(f"Skip empty file: {fpath.name}")
            continue

        cols = list(rows[0].keys())
        if TIME_COLUMN not in cols:
            raise ValueError(f"Saknar {TIME_COLUMN} i {fpath}")

        raw_str_data = {c: [row.get(c, "") for row in rows] for c in cols}

        t = np.array([safe_float(v) for v in raw_str_data[TIME_COLUMN]], dtype=np.float64)
        finite_t = np.isfinite(t)
        if np.count_nonzero(finite_t) < 2:
            raise ValueError(f"Ogiltig tidskolumn i {fpath}")

        dt = np.median(np.diff(t[finite_t]))
        fs = 1.0 / dt

        # Vi uppdaterar samma kolumnnamn
        updated_cols = {}

        for s in range(NUM_SENSORS):
            for axis in AXES:
                col = cal_col(s, axis)
                if col not in cols:
                    raise ValueError(f"Saknar kalibrerad kolumn {col} i {fpath.name}")

                x = np.array([safe_float(v) for v in raw_str_data[col]], dtype=np.float64)
                y = x.copy()

                # Mean removal först, om så valt
                if APPLY_MEAN_REMOVAL and not MEAN_REMOVAL_AFTER_HPF:
                    y = mean_remove(y)

                # HPF
                if APPLY_HPF:
                    y = butter_hpf_zero_phase(y, fs=fs, cutoff_hz=HPF_CUTOFF_HZ, order=HPF_ORDER)

                # Mean removal efter HPF, om så valt
                if APPLY_MEAN_REMOVAL and MEAN_REMOVAL_AFTER_HPF:
                    y = mean_remove(y)

                updated_cols[col] = y

        # Skriv output med samma kolumnnamn
        out_path = OUT_DIR / f"{fpath.stem}{OUTPUT_SUFFIX}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)

            for i in range(len(rows)):
                row = []
                for c in cols:
                    if c in updated_cols:
                        row.append(format_float(updated_cols[c][i], decimals=9))
                    else:
                        row.append(raw_str_data[c][i])
                w.writerow(row)

        print(f"OK: {fpath.name} -> {out_path}")

if __name__ == "__main__":
    main()