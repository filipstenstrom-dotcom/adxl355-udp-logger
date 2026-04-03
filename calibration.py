#!/usr/bin/env python3
import csv
import json
import glob
from pathlib import Path
import numpy as np

# =======================
# SETTINGS
# =======================

BASE_CAL_JSON = Path(r"G:/My Drive/Project SHM/Data/Info/cal_25C_3x3.json")
TEMP_MODEL_JSON = Path(r"G:/My Drive/Project SHM/Data/Info/TempCalibrationOut/temp_norm_model - Copy.json")
# Exempel:
# - temp_norm_model.json  (alpha_per_C)
# - temp_bias_fit.json    (k_g_per_C + c_g)

INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Impact/2026-04-03/*.csv"
INPUT_FILES = None

OUT_DIR = Path(r"G:/My Drive/Project SHM/Data/Impact/2026-04-03")

TIME_COLUMN = "recv_time_s"
A_PREFIX = "a"
N_A_COLS = 12
NUM_SENSORS = 2
AXES_PER_SENSOR = 3
LSB_PER_G = 256000.0

TEMP_COLS = ["temp0_C", "temp1_C", "temp2_C", "temp3_C"]

WRITE_RAW_G_COLUMNS = False
WRITE_BASE_CAL_COLUMNS = False
WRITE_TEMP_COLUMNS = False


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
    return f"temp{sensor_index}_C"

def format_float(v, decimals=9):
    if not np.isfinite(v):
        return ""
    return f"{v:.{decimals}f}"

def load_base_cal():
    base = json.loads(BASE_CAL_JSON.read_text(encoding="utf-8"))
    b0 = {}
    C = {}
    T0_base = {}

    for s in range(NUM_SENSORS):
        sb = base["sensors"][str(s)]
        b0[s] = np.array(sb["b0_g"], dtype=np.float64)
        C[s] = np.array(sb["C"], dtype=np.float64)

        if "file_temp_C_stats" not in sb or "mean" not in sb["file_temp_C_stats"]:
            raise ValueError(f"Saknar file_temp_C_stats.mean för sensor {s} i {BASE_CAL_JSON}")
        T0_base[s] = float(sb["file_temp_C_stats"]["mean"])

    return b0, C, T0_base

def load_temp_model():
    temp = json.loads(TEMP_MODEL_JSON.read_text(encoding="utf-8"))

    # detektera modelltyp från sensor 0
    s0 = temp["sensors"]["0"]

    if "alpha_per_C" in s0:
        mode = "norm_alpha"
    elif "k_g_per_C" in s0 and "c_g" in s0:
        mode = "post_cal_bias"
    else:
        raise ValueError("Okänd tempmodell-json. Förväntade alpha_per_C eller k_g_per_C + c_g.")

    model = {
        "mode": mode,
        "T0_temp": {},
        "alpha": {},
        "k": {},
        "c": {},
    }

    for s in range(NUM_SENSORS):
        st = temp["sensors"][str(s)]

        if "T0_C" not in st:
            raise ValueError(f"Saknar T0_C för sensor {s} i {TEMP_MODEL_JSON}")

        model["T0_temp"][s] = float(st["T0_C"])

        if mode == "norm_alpha":
            model["alpha"][s] = float(st["alpha_per_C"])
        elif mode == "post_cal_bias":
            model["k"][s] = np.array(st["k_g_per_C"], dtype=np.float64)
            model["c"][s] = np.array(st["c_g"], dtype=np.float64)

    return model

def load_raw_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if not rows:
        raise ValueError(f"CSV tom: {path}")

    cols = list(rows[0].keys())
    if TIME_COLUMN not in cols:
        raise ValueError(f"Saknar {TIME_COLUMN} i {path}")

    raw_str_data = {}
    for c in cols:
        raw_str_data[c] = [row.get(c, "") for row in rows]

    t = np.array([safe_float(v) for v in raw_str_data[TIME_COLUMN]], dtype=np.float64)

    a_counts = np.full((len(rows), N_A_COLS), np.nan, dtype=np.float64)
    for i in range(N_A_COLS):
        name = a_col(i)
        if name not in cols:
            raise ValueError(f"Saknar kolumn {name} i {path}")
        a_counts[:, i] = np.array([safe_float(v) for v in raw_str_data[name]], dtype=np.float64)

    temps = {}
    for tc in TEMP_COLS:
        if tc in cols:
            temps[tc] = np.array([safe_float(v) for v in raw_str_data[tc]], dtype=np.float64)

    return cols, raw_str_data, t, a_counts, temps


# =======================
# Main
# =======================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    b0, C, T0_base = load_base_cal()
    temp_model = load_temp_model()
    files = resolve_files()

    print(f"Found {len(files)} files.")
    print(f"Temp model mode: {temp_model['mode']}")

    for s in range(NUM_SENSORS):
        print(
            f"Sensor {s}: "
            f"T0_base={T0_base[s]:.6f} C, "
            f"T0_temp={temp_model['T0_temp'][s]:.6f} C"
        )

    for fpath in files:
        cols, raw_str_data, t, a_counts, temps = load_raw_csv(fpath)
        a_g = a_counts / float(LSB_PER_G)
        n = a_g.shape[0]

        new_cols = {}

        # raw g
        if WRITE_RAW_G_COLUMNS:
            for s in range(NUM_SENSORS):
                i0, i1, i2 = sensor_a_indices(s)
                new_cols[f"s{s}_ax_g_raw"] = a_g[:, i0].copy()
                new_cols[f"s{s}_ay_g_raw"] = a_g[:, i1].copy()
                new_cols[f"s{s}_az_g_raw"] = a_g[:, i2].copy()

        for s in range(NUM_SENSORS):
            i0, i1, i2 = sensor_a_indices(s)
            tcol = temp_col_for_sensor(s)
            if tcol not in temps:
                raise ValueError(f"Saknar temperaturkolumn {tcol} för sensor {s} i {fpath.name}")

            T = temps[tcol]

            meas = np.column_stack([a_g[:, i0], a_g[:, i1], a_g[:, i2]])   # (N,3)

            # Grundkalibrering (domän för b0 och C)
            base_cal = (meas - b0[s].reshape(1, 3)) @ C[s].T

            if WRITE_BASE_CAL_COLUMNS:
                new_cols[f"s{s}_ax_g_basecal"] = base_cal[:, 0]
                new_cols[f"s{s}_ay_g_basecal"] = base_cal[:, 1]
                new_cols[f"s{s}_az_g_basecal"] = base_cal[:, 2]

            # Temperaturmodell i rätt domän
            T0t = temp_model["T0_temp"][s]
            dT = T - T0t

            if temp_model["mode"] == "norm_alpha":
                alpha = temp_model["alpha"][s]
                scale = 1.0 + alpha * dT

                # skydd
                bad = ~np.isfinite(scale) | (np.abs(scale) < 1e-12)
                scale[bad] = np.nan

                final = base_cal / scale.reshape(-1, 1)

                if WRITE_TEMP_COLUMNS:
                    new_cols[f"s{s}_temp_scale"] = scale

            elif temp_model["mode"] == "post_cal_bias":
                k = temp_model["k"][s]
                c = temp_model["c"][s]
                bias_temp_post = c.reshape(1, 3) + dT.reshape(-1, 1) * k.reshape(1, 3)
                final = base_cal - bias_temp_post

                if WRITE_TEMP_COLUMNS:
                    new_cols[f"s{s}_bx_temp_post_g"] = bias_temp_post[:, 0]
                    new_cols[f"s{s}_by_temp_post_g"] = bias_temp_post[:, 1]
                    new_cols[f"s{s}_bz_temp_post_g"] = bias_temp_post[:, 2]

            else:
                raise ValueError("Okänt tempmodell-läge.")

            new_cols[f"s{s}_ax_g_cal"] = final[:, 0]
            new_cols[f"s{s}_ay_g_cal"] = final[:, 1]
            new_cols[f"s{s}_az_g_cal"] = final[:, 2]

        # header
        header = list(cols)
        extra_header = []

        if WRITE_RAW_G_COLUMNS:
            for s in range(NUM_SENSORS):
                extra_header += [
                    f"s{s}_ax_g_raw", f"s{s}_ay_g_raw", f"s{s}_az_g_raw"
                ]

        if WRITE_BASE_CAL_COLUMNS:
            for s in range(NUM_SENSORS):
                extra_header += [
                    f"s{s}_ax_g_basecal", f"s{s}_ay_g_basecal", f"s{s}_az_g_basecal"
                ]

        if WRITE_TEMP_COLUMNS:
            if temp_model["mode"] == "norm_alpha":
                for s in range(NUM_SENSORS):
                    extra_header += [f"s{s}_temp_scale"]
            elif temp_model["mode"] == "post_cal_bias":
                for s in range(NUM_SENSORS):
                    extra_header += [
                        f"s{s}_bx_temp_post_g",
                        f"s{s}_by_temp_post_g",
                        f"s{s}_bz_temp_post_g",
                    ]

        for s in range(NUM_SENSORS):
            extra_header += [
                f"s{s}_ax_g_cal", f"s{s}_ay_g_cal", f"s{s}_az_g_cal"
            ]

        header += extra_header

        out_path = OUT_DIR / f"{fpath.stem}_cal.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)

            for i in range(n):
                row = []
                for c in cols:
                    row.append(raw_str_data[c][i])
                for c in extra_header:
                    row.append(format_float(new_cols[c][i], decimals=9))
                w.writerow(row)

        print(f"OK: {fpath.name} -> {out_path}")

if __name__ == "__main__":
    main()