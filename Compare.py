#!/usr/bin/env python3
import csv
import re
import glob
from pathlib import Path
import numpy as np

# =======================
# SETTINGS
# =======================

INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Verification/*.csv"
# Eller:
# INPUT_FILES = [r"...file1.csv", r"...file2.csv"]
INPUT_FILES = None

OUT_DIR = Path(r"G:/My Drive/Project SHM/Data")
OUT_SUMMARY_CSV = OUT_DIR / "still_raw_vs_cal_summary.csv"

NUM_SENSORS = 2
AXES_PER_SENSOR = 3
LSB_PER_G = 256000.0

TIME_COLUMN = "recv_time_s"
A_PREFIX = "a"
N_A_COLS = 12

# för att ignorera början av filen om du vill
START_OFFSET_S = 0.0

# Filmatchning:
# vi matchar på denna del, t.ex. 20260308_133234
TIMESTAMP_REGEX = r"(\d{8}_\d{6})"

# CAL-kolumnnamn
# scriptet antar dessa extra kolumner i kalibrerad csv:
# s0_ax_g_cal, s0_ay_g_cal, s0_az_g_cal, ...
CAL_SUFFIX = "_cal"

# =======================
# Helpers
# =======================

def resolve_files():
    if INPUT_FILES is not None:
        return [Path(p) for p in INPUT_FILES]
    return [Path(p) for p in glob.glob(INPUT_GLOB, recursive=True)]

def safe_float(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    return float(s)

def extract_timestamp_key(filename: str):
    m = re.search(TIMESTAMP_REGEX, filename)
    return m.group(1) if m else None

def is_cal_file(path: Path):
    return "_cal" in path.stem.lower()

def a_col(i: int):
    return f"{A_PREFIX}{i}"

def sensor_a_indices(sensor_index: int):
    base = sensor_index * AXES_PER_SENSOR
    return base + 0, base + 1, base + 2

def cal_cols(sensor_index: int):
    return (
        f"s{sensor_index}_ax_g_cal",
        f"s{sensor_index}_ay_g_cal",
        f"s{sensor_index}_az_g_cal",
    )

def load_csv_dict(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"CSV tom: {path}")
    return rows, list(rows[0].keys())

def build_time_mask(rows, time_column, start_offset_s):
    if time_column not in rows[0]:
        return np.ones(len(rows), dtype=bool)

    t = np.array([safe_float(r.get(time_column, "")) for r in rows], dtype=np.float64)
    if not np.any(np.isfinite(t)):
        return np.ones(len(rows), dtype=bool)

    t0 = float(t[np.where(np.isfinite(t))[0][0]]) + start_offset_s
    return np.isfinite(t) & (t >= t0)

def get_raw_sensor_g(rows, sensor_index, mask):
    i0, i1, i2 = sensor_a_indices(sensor_index)
    x = np.array([safe_float(r.get(a_col(i0), "")) for r in rows], dtype=np.float64) / LSB_PER_G
    y = np.array([safe_float(r.get(a_col(i1), "")) for r in rows], dtype=np.float64) / LSB_PER_G
    z = np.array([safe_float(r.get(a_col(i2), "")) for r in rows], dtype=np.float64) / LSB_PER_G

    good = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(good) < 10:
        return None
    return x[good], y[good], z[good]

def get_cal_sensor_g(rows, sensor_index, mask):
    cx, cy, cz = cal_cols(sensor_index)
    x = np.array([safe_float(r.get(cx, "")) for r in rows], dtype=np.float64)
    y = np.array([safe_float(r.get(cy, "")) for r in rows], dtype=np.float64)
    z = np.array([safe_float(r.get(cz, "")) for r in rows], dtype=np.float64)

    good = mask & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(good) < 10:
        return None
    return x[good], y[good], z[good]

def compute_metrics(x, y, z):
    norm = np.sqrt(x*x + y*y + z*z)

    return {
        "count": int(len(norm)),
        "mean_norm": float(np.mean(norm)),
        "median_norm": float(np.median(norm)),
        "std_norm": float(np.std(norm, ddof=1)) if len(norm) > 1 else 0.0,
        "rms_norm_err_1g": float(np.sqrt(np.mean((norm - 1.0) ** 2))),
        "mean_x": float(np.mean(x)),
        "mean_y": float(np.mean(y)),
        "mean_z": float(np.mean(z)),
        "std_x": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "std_y": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "std_z": float(np.std(z, ddof=1)) if len(z) > 1 else 0.0,
    }

def improvement_percent(raw_val, cal_val):
    if not np.isfinite(raw_val) or raw_val <= 0:
        return np.nan
    return (raw_val - cal_val) / raw_val * 100.0

# =======================
# Main
# =======================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = resolve_files()
    if not files:
        raise ValueError("Inga filer hittades.")

    # gruppera på timestamp
    groups = {}
    for f in files:
        key = extract_timestamp_key(f.name)
        if key is None:
            continue
        groups.setdefault(key, {"raw": [], "cal": []})
        if is_cal_file(f):
            groups[key]["cal"].append(f)
        else:
            groups[key]["raw"].append(f)

    rows_out = []

    for key, grp in groups.items():
        if len(grp["raw"]) == 0 or len(grp["cal"]) == 0:
            continue

        # om flera kandidater finns tar vi första
        raw_path = sorted(grp["raw"])[0]
        cal_path = sorted(grp["cal"])[0]

        try:
            raw_rows, raw_cols = load_csv_dict(raw_path)
            cal_rows, cal_cols_list = load_csv_dict(cal_path)
        except Exception as e:
            print(f"Skip {key}: {e}")
            continue

        raw_mask = build_time_mask(raw_rows, TIME_COLUMN, START_OFFSET_S)
        cal_mask = build_time_mask(cal_rows, TIME_COLUMN, START_OFFSET_S)

        for s in range(NUM_SENSORS):
            raw_xyz = get_raw_sensor_g(raw_rows, s, raw_mask)
            cal_xyz = get_cal_sensor_g(cal_rows, s, cal_mask)

            if raw_xyz is None or cal_xyz is None:
                continue

            raw_metrics = compute_metrics(*raw_xyz)
            cal_metrics = compute_metrics(*cal_xyz)

            rows_out.append({
                "timestamp_key": key,
                "sensor": s,
                "raw_file": raw_path.name,
                "cal_file": cal_path.name,

                "raw_count": raw_metrics["count"],
                "cal_count": cal_metrics["count"],

                "raw_mean_norm": raw_metrics["mean_norm"],
                "cal_mean_norm": cal_metrics["mean_norm"],
                "raw_median_norm": raw_metrics["median_norm"],
                "cal_median_norm": cal_metrics["median_norm"],
                "raw_std_norm": raw_metrics["std_norm"],
                "cal_std_norm": cal_metrics["std_norm"],
                "raw_rms_norm_err_1g": raw_metrics["rms_norm_err_1g"],
                "cal_rms_norm_err_1g": cal_metrics["rms_norm_err_1g"],

                "improvement_rms_norm_err_percent": improvement_percent(
                    raw_metrics["rms_norm_err_1g"],
                    cal_metrics["rms_norm_err_1g"]
                ),

                "raw_mean_x": raw_metrics["mean_x"],
                "raw_mean_y": raw_metrics["mean_y"],
                "raw_mean_z": raw_metrics["mean_z"],
                "cal_mean_x": cal_metrics["mean_x"],
                "cal_mean_y": cal_metrics["mean_y"],
                "cal_mean_z": cal_metrics["mean_z"],

                "raw_std_x": raw_metrics["std_x"],
                "raw_std_y": raw_metrics["std_y"],
                "raw_std_z": raw_metrics["std_z"],
                "cal_std_x": cal_metrics["std_x"],
                "cal_std_y": cal_metrics["std_y"],
                "cal_std_z": cal_metrics["std_z"],
            })

    if not rows_out:
        raise ValueError("Inga matchade RAW/CAL-par hittades.")

    header = [
        "timestamp_key", "sensor", "raw_file", "cal_file",
        "raw_count", "cal_count",
        "raw_mean_norm", "cal_mean_norm",
        "raw_median_norm", "cal_median_norm",
        "raw_std_norm", "cal_std_norm",
        "raw_rms_norm_err_1g", "cal_rms_norm_err_1g",
        "improvement_rms_norm_err_percent",
        "raw_mean_x", "raw_mean_y", "raw_mean_z",
        "cal_mean_x", "cal_mean_y", "cal_mean_z",
        "raw_std_x", "raw_std_y", "raw_std_z",
        "cal_std_x", "cal_std_y", "cal_std_z",
    ]

    with OUT_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"Wrote: {OUT_SUMMARY_CSV}")
    print(f"Compared {len(rows_out)} RAW/CAL sensor-pairs.")

if __name__ == "__main__":
    main()