#!/usr/bin/env python3
import csv
import json
import glob
from pathlib import Path

import numpy as np

# =======================
# SETTINGS
# =======================

# Kalibreringsfil
CAL_JSON = Path("G:/My Drive/Project SHM/Data/Info/cal_25C_3x3.json")

# Input: alla csv i en mapp, eller ange lista manuellt
INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Temp_verification/*.csv"
INPUT_FILES = None   # ex: [r"G:/.../file1.csv", r"G:/.../file2.csv"]

# Output-mapp
OUT_DIR = Path("G:/My Drive/Project SHM/Data/Temp_verification/Calibrated_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Rå dataformat
A_PREFIX = "a"
N_A_COLS = 12

NUM_SENSORS = 2
AXES_PER_SENSOR = 3
LSB_PER_G = 256000.0

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

def a_col(i: int) -> str:
    return f"{A_PREFIX}{i}"

def sensor_a_indices(sensor_index: int):
    base = sensor_index * AXES_PER_SENSOR
    return (base + 0, base + 1, base + 2)

def resolve_files():
    if INPUT_FILES is not None:
        files = [Path(p) for p in INPUT_FILES]
    else:
        files = [Path(p) for p in sorted(glob.glob(INPUT_GLOB))]
    if not files:
        raise ValueError("Hittade inga CSV-filer. Kontrollera INPUT_GLOB / INPUT_FILES.")
    return files

def load_calibration(cal_json_path: Path):
    cal = json.loads(cal_json_path.read_text(encoding="utf-8"))

    b0 = {}
    C = {}

    for s in range(NUM_SENSORS):
        key = str(s)
        if key not in cal["sensors"]:
            raise ValueError(f"Kalibreringsfil saknar sensor {s}")
        b0[s] = np.array(cal["sensors"][key]["b0_g"], dtype=np.float64)
        C[s] = np.array(cal["sensors"][key]["C"], dtype=np.float64)

    return b0, C

# =======================
# Main processing
# =======================
def process_file(path: Path, b0, C):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_header = reader.fieldnames

    if not rows:
        print(f"Skip {path.name}: tom fil")
        return

    if original_header is None:
        print(f"Skip {path.name}: kunde inte läsa header")
        return

    # kontrollera att råa counts finns
    for i in range(N_A_COLS):
        col = a_col(i)
        if col not in original_header:
            raise ValueError(f"{path.name} saknar kolumn {col}")

    out_path = OUT_DIR / f"{path.stem}_cal.csv"

    # Ny header = original header + kalibrerade kolumner
    header = list(original_header)
    for s in range(NUM_SENSORS):
        header += [f"s{s}_ax_g", f"s{s}_ay_g", f"s{s}_az_g"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for row in rows:
            # börja med originalraden i exakt samma ordning som input
            out_row = [row.get(col, "") for col in original_header]

            # läs råa counts
            a_counts = np.full((N_A_COLS,), np.nan, dtype=np.float64)
            for i in range(N_A_COLS):
                a_counts[i] = safe_float(row.get(a_col(i), ""))

            # counts -> g
            a_g = a_counts / float(LSB_PER_G)

            # kalibrera rad för rad
            for s in range(NUM_SENSORS):
                i0, i1, i2 = sensor_a_indices(s)
                meas = np.array([a_g[i0], a_g[i1], a_g[i2]], dtype=np.float64)

                if np.all(np.isfinite(meas)):
                    cal_vec = C[s] @ (meas - b0[s])
                    out_row += [
                        f"{cal_vec[0]:.9f}",
                        f"{cal_vec[1]:.9f}",
                        f"{cal_vec[2]:.9f}",
                    ]
                else:
                    out_row += ["", "", ""]

            writer.writerow(out_row)

    print(f"OK: {path.name} -> {out_path.name}")

def main():
    files = resolve_files()
    b0, C = load_calibration(CAL_JSON)

    print(f"Loaded calibration from: {CAL_JSON}")
    print(f"Found {len(files)} file(s)")

    for path in files:
        process_file(path, b0, C)

if __name__ == "__main__":
    main()