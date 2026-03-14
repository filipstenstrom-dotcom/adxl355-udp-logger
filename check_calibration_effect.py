#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import numpy as np
import glob

# =======================
# SETTINGS
# =======================

CAL_JSON = Path(r"G:/My Drive/Project SHM/Data/Info/cal_25C_3x3_lstsq.json")

# Välj EN av dessa:

# 1️⃣ Läs alla CSV i mapp:
#INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Verification/*.csv"
INPUT_GLOB = None

# 2️⃣ Eller lista manuellt:
INPUT_FILES = [r"G:/My Drive/Project SHM/Data/Verification/frames_20260228_130822.csv"]
#INPUT_FILES = None#[r"...file1.csv", r"...file2.csv"]

TIME_COLUMN = "recv_time_s"
A_PREFIX = "a"
N_A_COLS = 12

NUM_SENSORS = 2
AXES_PER_SENSOR = 3

LSB_PER_G = 256000.0

START_OFFSET_S = 4

OUT_DIR = Path("G:/My Drive/Project SHM/Data/Info")
WRITE_REPORT_CSV = True

# =======================
# Helpers
# =======================

def resolve_files():
    if INPUT_FILES is not None:
        files = [Path(p) for p in INPUT_FILES]
    else:
        files = [Path(p) for p in sorted(glob.glob(INPUT_GLOB))]
    if not files:
        raise ValueError("Hittade inga CSV-filer. Kontrollera INPUT_GLOB eller INPUT_FILES.")
    return files


def safe_float(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    return float(s)


def a_col(i: int) -> str:
    return f"{A_PREFIX}{i}"


def sensor_a_indices(sensor_index: int):
    base = sensor_index * AXES_PER_SENSOR
    return (base + 0, base + 1, base + 2)


def load_raw_csv(path: Path):
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError(f"CSV tom: {path}")

    cols = rows[0].keys()
    if TIME_COLUMN not in cols:
        raise ValueError(f"Saknar {TIME_COLUMN} i {path}")

    t = np.array([safe_float(row[TIME_COLUMN]) for row in rows], dtype=np.float64)

    a = np.full((len(rows), N_A_COLS), np.nan, dtype=np.float64)
    for i in range(N_A_COLS):
        name = a_col(i)
        if name not in cols:
            raise ValueError(f"Saknar kolumn {name} i {path}")
        a[:, i] = np.array([safe_float(row.get(name, "")) for row in rows], dtype=np.float64)

    return t, a


def mean_point_over_file(t, ax, ay, az, t_start):
    mask = (t >= t_start) & np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    if np.count_nonzero(mask) < 10:
        return None
    return np.array([np.mean(ax[mask]), np.mean(ay[mask]), np.mean(az[mask])], dtype=np.float64)


def stats_of_norms(norms: np.ndarray):
    return {
        "count": int(norms.size),
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0,
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "rms_err_vs_1g": float(np.sqrt(np.mean((norms - 1.0) ** 2))),
        "max_abs_err_vs_1g": float(np.max(np.abs(norms - 1.0))),
    }


def improvement_percent(before_rms, after_rms):
    if before_rms <= 0:
        return 0.0
    return float((before_rms - after_rms) / before_rms * 100.0)


# =======================
# Main
# =======================

def main():
    files = resolve_files()
    print(f"Found {len(files)} CSV files.")

    cal = json.loads(CAL_JSON.read_text(encoding="utf-8"))

    b0 = {}
    C = {}
    for s in range(NUM_SENSORS):
        d = cal["sensors"][str(s)]
        b0[s] = np.array(d["b0_g"], dtype=np.float64)
        C[s] = np.array(d["C"], dtype=np.float64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    norms_raw_all = {s: [] for s in range(NUM_SENSORS)}
    norms_cal_all = {s: [] for s in range(NUM_SENSORS)}
    detail_rows = {s: [] for s in range(NUM_SENSORS)}

    used_files = 0

    for fpath in files:
        t, a_counts = load_raw_csv(fpath)
        a_g = a_counts / float(LSB_PER_G)

        t_start = float(t[0]) + START_OFFSET_S

        any_ok = False

        for s in range(NUM_SENSORS):
            i0, i1, i2 = sensor_a_indices(s)
            p = mean_point_over_file(t, a_g[:, i0], a_g[:, i1], a_g[:, i2], t_start)
            if p is None:
                continue

            if np.linalg.norm(p) < 0.1:
                continue

            r_raw = float(np.linalg.norm(p))
            p_cal = C[s] @ (p - b0[s])
            r_cal = float(np.linalg.norm(p_cal))

            norms_raw_all[s].append(r_raw)
            norms_cal_all[s].append(r_cal)

            detail_rows[s].append({
                "file": str(fpath.name),
                "raw_norm": r_raw,
                "cal_norm": r_cal
            })

            any_ok = True

        if any_ok:
            used_files += 1
            print(f"{fpath.name}: used")

    summary = {
        "used_files": int(used_files),
        "start_offset_s": float(START_OFFSET_S),
        "sensors": {}
    }

    for s in range(NUM_SENSORS):
        raw = np.array(norms_raw_all[s], dtype=np.float64)
        caln = np.array(norms_cal_all[s], dtype=np.float64)

        if raw.size < 1:
            print(f"Sensor {s}: too few files ({raw.size}).")
            continue

        raw_stats = stats_of_norms(raw)
        cal_stats = stats_of_norms(caln)

        summary["sensors"][str(s)] = {
            "raw": raw_stats,
            "calibrated": cal_stats,
            "improvement_rms_percent": improvement_percent(
                raw_stats["rms_err_vs_1g"], cal_stats["rms_err_vs_1g"]
            ),
        }

        print(f"\n=== Sensor {s} ===")
        print(f"RAW RMS err: {raw_stats['rms_err_vs_1g']:.6f}")
        print(f"CAL RMS err: {cal_stats['rms_err_vs_1g']:.6f}")
        print(f"Improvement: {summary['sensors'][str(s)]['improvement_rms_percent']:.1f}%")

    out_json = OUT_DIR / "calibration_effect_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {out_json}")


if __name__ == "__main__":
    main()