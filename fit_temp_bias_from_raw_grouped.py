#!/usr/bin/env python3
import csv
import json
import glob
from pathlib import Path
import numpy as np

# =======================
# SETTINGS
# =======================

# Din befintliga kalibrering utan temp-bias
CAL_JSON = Path(r"G:/My Drive/Project SHM/Data/Info/cal_25C_3x3.json")

# Rådatafiler
INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Temp_verification/*.csv"
INPUT_FILES = None  # eller lista med filer

TIME_COLUMN = "recv_time_s"
A_PREFIX = "a"
N_A_COLS = 12

NUM_SENSORS = 2
AXES_PER_SENSOR = 3
LSB_PER_G = 256000.0

START_OFFSET_S = 0.5

# temp-kolumner i råfilen
TEMP_COLS = ["temp0_C", "temp1_C", "temp2_C", "temp3_C"]

# riktningsgruppering
ANGLE_TOL_DEG = 2.0

# temperaturbinning
TEMP_BIN_MODE = "round"   # "round" | "floor" | "ceil"

# outlier removal på residual-vs-temp innan linjär fit
USE_OUTLIER_REJECTION = True
MAD_Z_THRESHOLD = 3.5

# output
OUT_DIR = Path(r"G:/My Drive/Project SHM/Data/Info/TempCalibrationOut")
OUT_JSON = OUT_DIR / "temp_bias_fit.json"
OUT_CSV_POINTS = OUT_DIR / "temp_bias_points.csv"

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

def a_col(i: int) -> str:
    return f"{A_PREFIX}{i}"

def sensor_a_indices(sensor_index: int):
    base = sensor_index * AXES_PER_SENSOR
    return (base + 0, base + 1, base + 2)

def temp_col_for_sensor(sensor_index: int):
    name = f"temp{sensor_index}_C"
    return name

def load_raw_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
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

    return t, a, temps

def temp_bin(T):
    if TEMP_BIN_MODE == "round":
        return int(np.round(T))
    if TEMP_BIN_MODE == "floor":
        return int(np.floor(T))
    if TEMP_BIN_MODE == "ceil":
        return int(np.ceil(T))
    raise ValueError("TEMP_BIN_MODE måste vara round/floor/ceil")

def mean_over_file(t, x, t_start):
    mask = (t >= t_start) & np.isfinite(x)
    if np.count_nonzero(mask) < 10:
        return np.nan
    return float(np.mean(x[mask]))

def mean_vector_over_file(t, ax, ay, az, t_start):
    mask = (t >= t_start) & np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    if np.count_nonzero(mask) < 10:
        return None
    return np.array([
        np.mean(ax[mask]),
        np.mean(ay[mask]),
        np.mean(az[mask])
    ], dtype=np.float64)

def normalize(v):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        return None
    return v / n

def angle_deg(u, v):
    uu = normalize(u)
    vv = normalize(v)
    if uu is None or vv is None:
        return np.inf
    c = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def robust_mask(y):
    """
    Returnerar mask för inliers via MAD-zscore.
    """
    y = np.asarray(y, dtype=np.float64)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad <= 0:
        return np.ones_like(y, dtype=bool)
    z = 0.6745 * (y - med) / mad
    return np.abs(z) <= MAD_Z_THRESHOLD

def fit_line(T, y):
    """
    Fit y = c + k*(T-T0)
    Returnerar T0, c, k, mask_used
    """
    T = np.asarray(T, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    good = np.isfinite(T) & np.isfinite(y)
    T = T[good]
    y = y[good]

    if T.size < 2:
        return np.nan, np.nan, np.nan, np.zeros_like(good, dtype=bool)

    if USE_OUTLIER_REJECTION and T.size >= 4:
        keep_local = robust_mask(y)
        T = T[keep_local]
        y = y[keep_local]
        if T.size < 2:
            return np.nan, np.nan, np.nan, np.zeros_like(good, dtype=bool)

    T0 = float(np.mean(T))
    X = np.column_stack([np.ones_like(T), T - T0])
    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    c = float(p[0])
    k = float(p[1])
    return T0, c, k, None

# =======================
# Direction grouping
# =======================
def build_direction_groups(rows_for_sensor, angle_tol_deg=1.0):
    """
    rows_for_sensor: list of dicts with 'dir_vec'
    Returnerar lista av grupper, varje grupp är lista av index.
    Enkel greedy clustering.
    """
    groups = []
    refs = []

    for idx, row in enumerate(rows_for_sensor):
        d = row["dir_vec"]
        assigned = False
        for gi, ref in enumerate(refs):
            if angle_deg(d, ref) <= angle_tol_deg:
                groups[gi].append(idx)
                # uppdatera referens som mean av gruppens dirs
                dirs = np.array([rows_for_sensor[j]["dir_vec"] for j in groups[gi]], dtype=np.float64)
                ref_new = normalize(np.mean(dirs, axis=0))
                if ref_new is not None:
                    refs[gi] = ref_new
                assigned = True
                break
        if not assigned:
            groups.append([idx])
            refs.append(normalize(d))

    return groups, refs

# =======================
# Main
# =======================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cal = json.loads(CAL_JSON.read_text(encoding="utf-8"))
    b0 = {}
    C = {}
    for s in range(NUM_SENSORS):
        d = cal["sensors"][str(s)]
        b0[s] = np.array(d["b0_g"], dtype=np.float64)
        C[s] = np.array(d["C"], dtype=np.float64)

    files = resolve_files()
    print(f"Found {len(files)} files")

    # en rad per fil per sensor
    rows_by_sensor = {s: [] for s in range(NUM_SENSORS)}

    for fpath in files:
        t, a_counts, temps = load_raw_csv(fpath)
        a_g = a_counts / float(LSB_PER_G)
        t_start = float(t[0]) + START_OFFSET_S

        for s in range(NUM_SENSORS):
            i0, i1, i2 = sensor_a_indices(s)
            mean_raw = mean_vector_over_file(t, a_g[:, i0], a_g[:, i1], a_g[:, i2], t_start)
            if mean_raw is None:
                continue
            if np.linalg.norm(mean_raw) < 0.1:
                continue

            # grundkalibrering utan temp-bias
            mean_cal = C[s] @ (mean_raw - b0[s])

            dir_vec = normalize(mean_cal)
            if dir_vec is None:
                continue

            tcol = temp_col_for_sensor(s)
            if tcol not in temps:
                continue
            Tm = mean_over_file(t, temps[tcol], t_start)
            if not np.isfinite(Tm):
                continue

            rows_by_sensor[s].append({
                "file": fpath.name,
                "temp_C": float(Tm),
                "temp_bin_C": temp_bin(Tm),
                "mean_raw": mean_raw,
                "mean_cal": mean_cal,
                "dir_vec": dir_vec,
            })

    # gruppera riktningar och skapa residualpunkter
    out = {
        "cal_json": str(CAL_JSON),
        "input_glob": INPUT_GLOB if INPUT_FILES is None else None,
        "start_offset_s": float(START_OFFSET_S),
        "angle_tol_deg": float(ANGLE_TOL_DEG),
        "temp_bin_mode": TEMP_BIN_MODE,
        "sensors": {}
    }

    csv_rows = []

    for s in range(NUM_SENSORS):
        rows = rows_by_sensor[s]
        if len(rows) < 4:
            print(f"Sensor {s}: too few rows")
            continue

        groups, refs = build_direction_groups(rows, ANGLE_TOL_DEG)

        # referensvektor per grupp
        group_ref = {}
        for gi, idxs in enumerate(groups):
            dirs = np.array([rows[j]["dir_vec"] for j in idxs], dtype=np.float64)
            ref = normalize(np.mean(dirs, axis=0))
            if ref is None:
                continue
            group_ref[gi] = ref

        # tilldela grupp-id
        row_to_group = {}
        for gi, idxs in enumerate(groups):
            for j in idxs:
                row_to_group[j] = gi

        # residual per fil
        residual_rows = []
        for j, row in enumerate(rows):
            gi = row_to_group[j]
            if gi not in group_ref:
                continue
            gref = group_ref[gi]
            resid = row["mean_cal"] - gref

            residual_rows.append({
                "file": row["file"],
                "group_id": gi,
                "temp_C": row["temp_C"],
                "temp_bin_C": row["temp_bin_C"],
                "mean_cal_x": row["mean_cal"][0],
                "mean_cal_y": row["mean_cal"][1],
                "mean_cal_z": row["mean_cal"][2],
                "gref_x": gref[0],
                "gref_y": gref[1],
                "gref_z": gref[2],
                "resid_x": resid[0],
                "resid_y": resid[1],
                "resid_z": resid[2],
            })

        # temperaturbinning inom varje grupp
        binned_rows = []
        keyset = {}
        for r in residual_rows:
            key = (r["group_id"], r["temp_bin_C"])
            keyset.setdefault(key, []).append(r)

        for (gi, Tb), rr in keyset.items():
            arrx = np.array([x["resid_x"] for x in rr], dtype=np.float64)
            arry = np.array([x["resid_y"] for x in rr], dtype=np.float64)
            arrz = np.array([x["resid_z"] for x in rr], dtype=np.float64)
            arrT = np.array([x["temp_C"] for x in rr], dtype=np.float64)
            binned_rows.append({
                "group_id": gi,
                "temp_bin_C": Tb,
                "temp_C_mean": float(np.mean(arrT)),
                "resid_x": float(np.mean(arrx)),
                "resid_y": float(np.mean(arry)),
                "resid_z": float(np.mean(arrz)),
                "count": int(len(rr)),
            })

        # fit kx, ky, kz på alla grupper/bin tillsammans
        T_all = np.array([r["temp_C_mean"] for r in binned_rows], dtype=np.float64)
        rx_all = np.array([r["resid_x"] for r in binned_rows], dtype=np.float64)
        ry_all = np.array([r["resid_y"] for r in binned_rows], dtype=np.float64)
        rz_all = np.array([r["resid_z"] for r in binned_rows], dtype=np.float64)

        T0x, cx, kx, _ = fit_line(T_all, rx_all)
        T0y, cy, ky, _ = fit_line(T_all, ry_all)
        T0z, cz, kz, _ = fit_line(T_all, rz_all)

        T0 = np.nanmean([T0x, T0y, T0z])

        out["sensors"][str(s)] = {
            "num_files": int(len(rows)),
            "num_direction_groups": int(len(groups)),
            "num_binned_points": int(len(binned_rows)),
            "T0_C": float(T0),
            "k_g_per_C": [float(kx), float(ky), float(kz)],
            "c_g": [float(cx), float(cy), float(cz)],
        }

        for r in binned_rows:
            csv_rows.append([
                s,
                r["group_id"],
                r["temp_bin_C"],
                f"{r['temp_C_mean']:.3f}",
                f"{r['resid_x']:.9f}",
                f"{r['resid_y']:.9f}",
                f"{r['resid_z']:.9f}",
                r["count"],
            ])

        print(f"\n=== Sensor {s} ===")
        print(f"files: {len(rows)}")
        print(f"direction groups: {len(groups)}")
        print(f"binned points: {len(binned_rows)}")
        print(f"T0 = {T0:.3f} C")
        print(f"kx, ky, kz = {kx:.9e}, {ky:.9e}, {kz:.9e} g/C")

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    with OUT_CSV_POINTS.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "group_id", "temp_bin_C", "temp_C_mean", "resid_x", "resid_y", "resid_z", "count"])
        w.writerows(csv_rows)

    print(f"\nWrote: {OUT_JSON}")
    print(f"Wrote: {OUT_CSV_POINTS}")

if __name__ == "__main__":
    main()