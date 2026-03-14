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

# 1) Alla CSV i en mapp:
INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Battery_verification/*.csv"
# 2) Eller lista manuellt:
INPUT_FILES = None  # [r"...file1.csv", r"...file2.csv"]

TIME_COLUMN = "recv_time_s"
A_PREFIX = "a"
N_A_COLS = 12

NUM_SENSORS = 2
AXES_PER_SENSOR = 3

LSB_PER_G = 256000.0
START_OFFSET_S = 0.0

# LPF-inställningar (för statisk verifiering)
LPF_CUTOFF_HZ = 0.5       # 0.3–0.8 Hz är vanligt för gravitationstest
LPF_MIN_DT_S = 1e-6       # skydd mot dt=0
LPF_MAX_DT_S = 1.0        # skydd mot konstiga timestamps

OUT_DIR = Path(r"G:/My Drive/Project SHM/Data/Verification")
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

def one_pole_lpf(t: np.ndarray, x: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """
    1:a ordningens IIR lågpass (time-varying dt):
      y[n] = y[n-1] + alpha*(x[n] - y[n-1])
    alpha = 1 - exp(-2*pi*fc*dt)
    Hanterar NaN genom att "hålla senaste y" tills giltigt x återkommer.
    """
    y = np.full_like(x, np.nan, dtype=np.float64)
    if x.size == 0:
        return y

    # starta på första giltiga sample
    idx0 = np.where(np.isfinite(x) & np.isfinite(t))[0]
    if idx0.size == 0:
        return y
    i0 = int(idx0[0])
    y_prev = float(x[i0])
    y[i0] = y_prev

    w = 2.0 * np.pi * float(cutoff_hz)

    for i in range(i0 + 1, x.size):
        if not (np.isfinite(t[i]) and np.isfinite(t[i-1])):
            y[i] = y_prev
            continue
        dt = float(t[i] - t[i-1])
        if dt < LPF_MIN_DT_S:
            dt = LPF_MIN_DT_S
        if dt > LPF_MAX_DT_S:
            dt = LPF_MAX_DT_S

        if not np.isfinite(x[i]):
            # håll senaste värde om x är NaN/inf
            y[i] = y_prev
            continue

        alpha = 1.0 - np.exp(-w * dt)
        y_prev = y_prev + alpha * (float(x[i]) - y_prev)
        y[i] = y_prev

    return y

def window_mask(t: np.ndarray, t_start: float):
    return (t >= t_start) & np.isfinite(t)

def lpf_metrics(t, ax, ay, az, t_start):
    """
    Returnerar LPF-baserade metrik för statisk data:
      - mean vector (LPF)
      - leakage_xy (LPF mean)
      - rms_err_norm_1g på LPF norm
    """
    m = window_mask(t, t_start) & np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    if np.count_nonzero(m) < 10:
        return None

    tt = t[m]
    x = ax[m]; y = ay[m]; z = az[m]

    xlp = one_pole_lpf(tt, x, LPF_CUTOFF_HZ)
    ylp = one_pole_lpf(tt, y, LPF_CUTOFF_HZ)
    zlp = one_pole_lpf(tt, z, LPF_CUTOFF_HZ)

    mm = np.isfinite(xlp) & np.isfinite(ylp) & np.isfinite(zlp)
    if np.count_nonzero(mm) < 10:
        return None

    xlp = xlp[mm]; ylp = ylp[mm]; zlp = zlp[mm]

    mean_vec = np.array([np.mean(xlp), np.mean(ylp), np.mean(zlp)], dtype=np.float64)
    leak_xy = float(np.sqrt(mean_vec[0]**2 + mean_vec[1]**2))

    norm_lp = np.sqrt(xlp**2 + ylp**2 + zlp**2)
    rms_err = float(np.sqrt(np.mean((norm_lp - 1.0)**2)))
    max_abs_err = float(np.max(np.abs(norm_lp - 1.0)))

    return {
        "mean_x": float(mean_vec[0]),
        "mean_y": float(mean_vec[1]),
        "mean_z": float(mean_vec[2]),
        "leak_xy": leak_xy,
        "rms_err_norm_1g": rms_err,
        "max_abs_err_norm_1g": max_abs_err,
    }

def stats(arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else np.nan,
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)) if arr.size else np.nan,
        "max": float(np.max(arr)) if arr.size else np.nan,
    }

def improvement_percent(before, after):
    if before <= 0 or not np.isfinite(before) or not np.isfinite(after):
        return np.nan
    return float((before - after) / before * 100.0)

# =======================
# Main
# =======================
def main():
    files = resolve_files()
    print(f"Found {len(files)} CSV files.")

    cal = json.loads(CAL_JSON.read_text(encoding="utf-8"))
    if "sensors" not in cal:
        raise ValueError("CAL_JSON saknar 'sensors'.")

    b0 = {}
    C = {}
    for s in range(NUM_SENSORS):
        d = cal["sensors"][str(s)]
        b0[s] = np.array(d["b0_g"], dtype=np.float64)
        C[s] = np.array(d["C"], dtype=np.float64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # samlingslistor per sensor (för summary)
    raw_rms = {s: [] for s in range(NUM_SENSORS)}
    cal_rms = {s: [] for s in range(NUM_SENSORS)}
    raw_leak = {s: [] for s in range(NUM_SENSORS)}
    cal_leak = {s: [] for s in range(NUM_SENSORS)}

    detail_rows = {s: [] for s in range(NUM_SENSORS)}

    used_files = 0

    for fpath in files:
        t, a_counts = load_raw_csv(fpath)
        a_g = a_counts / float(LSB_PER_G)
        t_start = float(t[0]) + START_OFFSET_S

        any_ok = False
        for s in range(NUM_SENSORS):
            i0, i1, i2 = sensor_a_indices(s)
            ax = a_g[:, i0]; ay = a_g[:, i1]; az = a_g[:, i2]

            # RAW LPF metrics
            m_raw = lpf_metrics(t, ax, ay, az, t_start)
            if m_raw is None:
                continue

            # CAL: applicera per-sample först (linjärt), sen LPF metrics
            # a_cal(t) = C @ (a_raw(t) - b0)
            X = np.vstack([ax, ay, az]).T  # (N,3)
            # hantera NaN: maska innan transform?
            # vi låter NaN passera (ger NaN i resultat), LPF-funktionen hanterar det.
            Xc = (X - b0[s].reshape(1, 3)) @ C[s].T  # (N,3)
            m_cal = lpf_metrics(t, Xc[:,0], Xc[:,1], Xc[:,2], t_start)
            if m_cal is None:
                continue

            raw_rms[s].append(m_raw["rms_err_norm_1g"])
            cal_rms[s].append(m_cal["rms_err_norm_1g"])
            raw_leak[s].append(m_raw["leak_xy"])
            cal_leak[s].append(m_cal["leak_xy"])

            detail_rows[s].append({
                "file": fpath.name,
                "raw_mean_x": m_raw["mean_x"],
                "raw_mean_y": m_raw["mean_y"],
                "raw_mean_z": m_raw["mean_z"],
                "raw_leak_xy": m_raw["leak_xy"],
                "raw_rms_norm_err": m_raw["rms_err_norm_1g"],
                "cal_mean_x": m_cal["mean_x"],
                "cal_mean_y": m_cal["mean_y"],
                "cal_mean_z": m_cal["mean_z"],
                "cal_leak_xy": m_cal["leak_xy"],
                "cal_rms_norm_err": m_cal["rms_err_norm_1g"],
            })

            any_ok = True

        if any_ok:
            used_files += 1
            print(f"{fpath.name}: used")
        else:
            print(f"{fpath.name}: skipped (no valid LPF metrics)")

    summary = {
        "cal_json": str(CAL_JSON),
        "input_glob": INPUT_GLOB if INPUT_FILES is None else None,
        "used_files": int(used_files),
        "lsb_per_g": float(LSB_PER_G),
        "start_offset_s": float(START_OFFSET_S),
        "lpf_cutoff_hz": float(LPF_CUTOFF_HZ),
        "sensors": {}
    }

    for s in range(NUM_SENSORS):
        rr = np.array(raw_rms[s], dtype=np.float64)
        cr = np.array(cal_rms[s], dtype=np.float64)
        rl = np.array(raw_leak[s], dtype=np.float64)
        cl = np.array(cal_leak[s], dtype=np.float64)

        if rr.size < 1:
            print(f"Sensor {s}: too few files ({rr.size}).")
            continue

        rms_raw_stats = stats(rr)
        rms_cal_stats = stats(cr)
        leak_raw_stats = stats(rl)
        leak_cal_stats = stats(cl)

        summary["sensors"][str(s)] = {
            "rms_norm_err_raw": rms_raw_stats,
            "rms_norm_err_cal": rms_cal_stats,
            "rms_norm_err_improvement_percent": improvement_percent(rms_raw_stats["mean"], rms_cal_stats["mean"]),
            "leak_xy_raw": leak_raw_stats,
            "leak_xy_cal": leak_cal_stats,
            "leak_xy_improvement_percent": improvement_percent(leak_raw_stats["mean"], leak_cal_stats["mean"]),
        }

        print(f"\n=== Sensor {s} (LPF verification) ===")
        print(f"RMS(norm-1g) RAW mean={rms_raw_stats['mean']:.6e}  CAL mean={rms_cal_stats['mean']:.6e}  "
              f"Improvement={summary['sensors'][str(s)]['rms_norm_err_improvement_percent']:.1f}%")
        print(f"LeakXY (mean LPF) RAW mean={leak_raw_stats['mean']:.6e}  CAL mean={leak_cal_stats['mean']:.6e}  "
              f"Improvement={summary['sensors'][str(s)]['leak_xy_improvement_percent']:.1f}%")

        if WRITE_REPORT_CSV:
            out_csv = OUT_DIR / f"s{s}_lpf_report.csv"
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "file",
                    "raw_mean_x","raw_mean_y","raw_mean_z","raw_leak_xy","raw_rms_norm_err",
                    "cal_mean_x","cal_mean_y","cal_mean_z","cal_leak_xy","cal_rms_norm_err",
                ])
                for row in detail_rows[s]:
                    w.writerow([
                        row["file"],
                        f"{row['raw_mean_x']:.9f}", f"{row['raw_mean_y']:.9f}", f"{row['raw_mean_z']:.9f}",
                        f"{row['raw_leak_xy']:.9f}", f"{row['raw_rms_norm_err']:.9e}",
                        f"{row['cal_mean_x']:.9f}", f"{row['cal_mean_y']:.9f}", f"{row['cal_mean_z']:.9f}",
                        f"{row['cal_leak_xy']:.9f}", f"{row['cal_rms_norm_err']:.9e}",
                    ])
            print(f"Wrote: {out_csv}")

    out_json = OUT_DIR / "calibration_effect_summary_lpf.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {out_json}")

if __name__ == "__main__":
    main()