#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import numpy as np

# ===== plotting (optional) =====
PLOT_3D = True
PLOT_OUTDIR = Path("G:/My Drive/Project SHM/Data/Info")
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    plt = None
    PLOT_3D = False

# =======================
# SETTINGS
# =======================

INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Calibration/*.csv"
INPUT_FILES = None  # ex: [r".../run1.csv", r".../run2.csv"]

TIME_COLUMN = "recv_time_s"
A_PREFIX = "a"
N_A_COLS = 12

NUM_SENSORS = 2
AXES_PER_SENSOR = 3
LSB_PER_G = 256000.0

START_OFFSET_S = 0.0

OUTPUT_JSON = Path("G:/My Drive/Project SHM/Data/Info/cal_25C_diag.json")

TEMP_COLS = ["temp0_C", "temp1_C", "temp2_C", "temp3_C"]
SAVE_TEMP_STATS = True

# =======================
# Helpers
# =======================
def safe_float(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    return float(s)

def a_col(idx: int) -> str:
    return f"{A_PREFIX}{idx}"

def sensor_a_indices(sensor_index: int):
    base = sensor_index * AXES_PER_SENSOR
    return (base + 0, base + 1, base + 2)

def resolve_files():
    if INPUT_FILES is not None:
        files = [Path(p) for p in INPUT_FILES]
    else:
        import glob
        files = [Path(p) for p in sorted(glob.glob(INPUT_GLOB))]
    if not files:
        raise ValueError("Hittade inga CSV-filer. Kolla INPUT_GLOB/INPUT_FILES.")
    return files

def load_csv(path: Path):
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

    temps = {}
    for tn in TEMP_COLS:
        if tn in cols:
            temps[tn] = np.array([safe_float(row.get(tn, "")) for row in rows], dtype=np.float64)

    return {"t": t, "a_counts": a, "temps": temps}

def mean_point_over_file(t, ax, ay, az, t_start):
    mask = (t >= t_start) & np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    if np.count_nonzero(mask) < 10:
        return None
    return np.array([np.mean(ax[mask]), np.mean(ay[mask]), np.mean(az[mask])], dtype=np.float64)

# =======================
# Ellipsoid fit -> center b0 + full C
# =======================
def fit_ellipsoid_center_matrix(P: np.ndarray):
    if P.shape[0] < 10:
        raise ValueError("För få punkter för ellipsoid-fit (behöver helst 12+).")

    x = P[:, 0]
    y = P[:, 1]
    z = P[:, 2]

    D = np.column_stack([
        x * x, y * y, z * z,
        2 * x * y, 2 * x * z, 2 * y * z,
        2 * x, 2 * y, 2 * z
    ])
    rhs = np.ones((P.shape[0],), dtype=np.float64)
    p, *_ = np.linalg.lstsq(D, rhs, rcond=None)

    A = np.array([
        [p[0], p[3], p[4]],
        [p[3], p[1], p[5]],
        [p[4], p[5], p[2]],
    ], dtype=np.float64)
    A = 0.5 * (A + A.T)

    b = np.array([p[6], p[7], p[8]], dtype=np.float64)
    c = -1.0

    center = -0.5 * np.linalg.solve(A, b)
    k = float(center.T @ A @ center - c)
    if k <= 0 or not np.isfinite(k):
        raise ValueError("Ogiltig normalisering (k<=0). Datan kan vara dåligt spridd eller för bullrig.")

    Q = A / k
    try:
        C_full = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Q)
        if np.any(w <= 0):
            raise ValueError("Q är inte positiv definit. Datan kan vara för dåligt spridd eller för bullrig.")
        C_full = (V * np.sqrt(w)) @ V.T

    return center, C_full

# =======================
# Plotting
# =======================
def plot_points_3d(P, title, out_path):
    if not PLOT_3D or plt is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=12)
    ax.set_title(title)
    ax.set_xlabel("X [g]")
    ax.set_ylabel("Y [g]")
    ax.set_zlabel("Z [g]")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def plot_calibrated_on_sphere(P_cal, title, out_path):
    if not PLOT_3D or plt is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, linewidth=0.4, alpha=0.5)
    ax.scatter(P_cal[:, 0], P_cal[:, 1], P_cal[:, 2], s=12)
    ax.set_title(title)
    ax.set_xlabel("X [g]")
    ax.set_ylabel("Y [g]")
    ax.set_zlabel("Z [g]")
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def norm_stats(P_cal: np.ndarray):
    norms = np.linalg.norm(P_cal, axis=1)
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms, ddof=1)) if norms.size > 1 else 0.0,
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "count": int(norms.size),
        "rms_err_vs_1g": float(np.sqrt(np.mean((norms - 1.0) ** 2))),
        "max_abs_err_vs_1g": float(np.max(np.abs(norms - 1.0))),
    }

# =======================
# Main
# =======================
def main():
    files = resolve_files()
    print(f"Found {len(files)} CSV files.")

    points_g = {s: [] for s in range(NUM_SENSORS)}
    file_temp_stats = {s: [] for s in range(NUM_SENSORS)}
    used_files = 0

    for fpath in files:
        data = load_csv(fpath)
        t = data["t"]
        a_counts = data["a_counts"]
        temps = data["temps"]

        a_g = a_counts / float(LSB_PER_G)
        t_start = float(t[0]) + START_OFFSET_S

        ok_any = False
        for s in range(NUM_SENSORS):
            i0, i1, i2 = sensor_a_indices(s)
            p = mean_point_over_file(t, a_g[:, i0], a_g[:, i1], a_g[:, i2], t_start)
            if p is None:
                continue
            if np.linalg.norm(p) < 0.1:
                continue

            points_g[s].append(p)
            ok_any = True

            if SAVE_TEMP_STATS and temps:
                temp_name = f"temp{s}_C"
                if temp_name in temps:
                    maskT = (t >= t_start) & np.isfinite(temps[temp_name])
                    if np.count_nonzero(maskT) > 5:
                        file_temp_stats[s].append(float(np.mean(temps[temp_name][maskT])))

        if ok_any:
            used_files += 1
            print(f"{fpath.name}: used as 1 calibration point")
        else:
            print(f"{fpath.name}: skipped (no valid point)")

    out = {
        "input_files": [str(p) for p in files],
        "used_files": int(used_files),
        "method": "DIAG calibration only: C = diag(diag(C_full)); points are 1 mean per file after START_OFFSET_S",
        "lsb_per_g": float(LSB_PER_G),
        "num_sensors": int(NUM_SENSORS),
        "start_offset_s": float(START_OFFSET_S),
        "sensors": {}
    }

    for s in range(NUM_SENSORS):
        P = np.array(points_g[s], dtype=np.float64)
        if P.shape[0] < 10:
            raise ValueError(f"Sensor {s}: för få giltiga punkter ({P.shape[0]}). Behöver helst 12+ filer.")

        plot_points_3d(P, f"Sensor {s} raw mean points (g)", PLOT_OUTDIR / f"s{s}_raw_points.png")

        b0, C_full = fit_ellipsoid_center_matrix(P)
        C = np.diag(np.diag(C_full))  # <-- DIAG only

        P_cal = (P - b0) @ C.T
        stats = norm_stats(P_cal)

        plot_calibrated_on_sphere(
            P_cal, f"Sensor {s} DIAG calibrated points (unit sphere target)",
            PLOT_OUTDIR / f"s{s}_cal_diag_points.png"
        )

        sensor_out = {
            "b0_g": b0.tolist(),
            "C": C.tolist(),
            "norm_stats": stats,
        }

        if SAVE_TEMP_STATS and file_temp_stats[s]:
            Ts = np.array(file_temp_stats[s], dtype=np.float64)
            sensor_out["file_temp_C_stats"] = {
                "mean": float(np.mean(Ts)),
                "min": float(np.min(Ts)),
                "max": float(np.max(Ts)),
                "count": int(Ts.size),
            }

        out["sensors"][str(s)] = sensor_out

        print(f"\n=== Sensor {s} (DIAG) ===")
        print(f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
              f"rms_err={stats['rms_err_vs_1g']:.6f} max_abs_err={stats['max_abs_err_vs_1g']:.6f}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote: {OUTPUT_JSON}")
    if PLOT_3D:
        print(f"Wrote plots to: {PLOT_OUTDIR.resolve()}")
    print("Apply per sample: a_cal = C @ (a_meas_g - b0)   (DIAG only)")

if __name__ == "__main__":
    main()