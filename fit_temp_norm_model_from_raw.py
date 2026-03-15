#!/usr/bin/env python3
import csv
import json
import glob
from pathlib import Path
import numpy as np

# =======================
# SETTINGS
# =======================

CAL_JSON = Path(r"G:/My Drive/Project SHM/Data/Info/cal_25C_3x3.json")

INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Temp_verification/*.csv"

INPUT_FILES = None

TIME_COLUMN = "recv_time_s"
A_PREFIX = "a"
N_A_COLS = 12

NUM_SENSORS = 2
AXES_PER_SENSOR = 3
LSB_PER_G = 256000.0

START_OFFSET_S = 0.5

TEMP_COLS = ["temp0_C", "temp1_C", "temp2_C", "temp3_C"]

OUT_DIR = Path(r"G:/My Drive/Project SHM/Data/Info/TempCalibrationOut")
OUT_JSON = OUT_DIR / "temp_norm_model.json"
OUT_POINTS = OUT_DIR / "temp_norm_points.csv"

USE_OUTLIER_REJECTION = True
MAD_Z = 3

# =======================
# Helpers
# =======================

def resolve_files():
    if INPUT_FILES is not None:
        return [Path(p) for p in INPUT_FILES]
    return [Path(p) for p in sorted(glob.glob(INPUT_GLOB))]

def safe_float(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    return float(s)

def a_col(i):
    return f"{A_PREFIX}{i}"

def sensor_a_indices(s):
    base = s * AXES_PER_SENSOR
    return base, base + 1, base + 2

def temp_col_for_sensor(s):
    return f"temp{s}_C"

def load_raw_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)

    t = np.array([safe_float(r[TIME_COLUMN]) for r in rows], dtype=np.float64)

    a = np.full((len(rows), N_A_COLS), np.nan)
    for i in range(N_A_COLS):
        name = a_col(i)
        a[:, i] = [safe_float(r.get(name, "")) for r in rows]

    temps = {}
    for tc in TEMP_COLS:
        if tc in rows[0]:
            temps[tc] = np.array([safe_float(r.get(tc, "")) for r in rows])

    return t, a, temps

def mean_vector(t, x, y, z, t_start):
    mask = (t >= t_start) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(mask) < 10:
        return None
    return np.array([
        np.mean(x[mask]),
        np.mean(y[mask]),
        np.mean(z[mask])
    ])

def mean_scalar(t, x, t_start):
    mask = (t >= t_start) & np.isfinite(x)
    if np.count_nonzero(mask) < 10:
        return np.nan
    return float(np.mean(x[mask]))

def robust_mask(y):
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad <= 0:
        return np.ones_like(y, bool)
    z = 0.6745 * (y - med) / mad
    return np.abs(z) <= MAD_Z

def fit_alpha(T, n):
    good = np.isfinite(T) & np.isfinite(n)
    T = T[good]
    n = n[good]

    if USE_OUTLIER_REJECTION and len(n) >= 5:
        keep = robust_mask(n)
        T = T[keep]
        n = n[keep]

    if len(T) < 2:
        return np.nan, np.nan

    T0 = np.mean(T)
    X = np.column_stack([np.ones_like(T), T - T0])
    p, *_ = np.linalg.lstsq(X, n - 1.0, rcond=None)

    c = p[0]
    alpha = p[1]
    return T0, alpha

# =======================
# Main
# =======================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cal = json.loads(CAL_JSON.read_text())

    b0 = {}
    C = {}
    for s in range(NUM_SENSORS):
        d = cal["sensors"][str(s)]
        b0[s] = np.array(d["b0_g"])
        C[s] = np.array(d["C"])

    files = resolve_files()
    print(f"Found {len(files)} files")

    rows_out = []
    model = {"sensors": {}}

    for s in range(NUM_SENSORS):
        T_list = []
        n_list = []

        for fpath in files:
            t, a_counts, temps = load_raw_csv(fpath)
            a_g = a_counts / LSB_PER_G

            t_start = t[0] + START_OFFSET_S

            i0, i1, i2 = sensor_a_indices(s)
            mean_raw = mean_vector(
                t,
                a_g[:, i0],
                a_g[:, i1],
                a_g[:, i2],
                t_start
            )
            if mean_raw is None:
                continue

            mean_cal = C[s] @ (mean_raw - b0[s])
            norm = np.linalg.norm(mean_cal)

            tcol = temp_col_for_sensor(s)
            if tcol not in temps:
                continue

            Tm = mean_scalar(t, temps[tcol], t_start)
            if not np.isfinite(Tm):
                continue

            T_list.append(Tm)
            n_list.append(norm)

            rows_out.append([s, fpath.name, Tm, norm])

        T_arr = np.array(T_list)
        n_arr = np.array(n_list)

        T0, alpha = fit_alpha(T_arr, n_arr)

        model["sensors"][str(s)] = {
            "T0_C": float(T0),
            "alpha_per_C": float(alpha),
            "num_points": int(len(T_arr)),
        }

        print(f"\nSensor {s}")
        print(f"T0 = {T0:.3f} C")
        print(f"alpha = {alpha:.6e} per C")

    OUT_JSON.write_text(json.dumps(model, indent=2))

    with OUT_POINTS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "file", "temp_C", "norm"])
        w.writerows(rows_out)

    print(f"\nWrote: {OUT_JSON}")
    print(f"Wrote: {OUT_POINTS}")

if __name__ == "__main__":
    main()