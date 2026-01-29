#!/usr/bin/env python3
import csv
import json
import struct
from pathlib import Path

import numpy as np

# =======================
# SETTINGS (ändra här)
# =======================

# Input: .bin (rå frames) eller .csv (från din logger)
INPUT_PATH   = Path(r"G:\My Drive\Project SHM\Data\frames_20260127_112812.csv")
INPUT_FORMAT = "auto"   # "auto" | "bin" | "csv"

# Output: "csv" för debug, "npz" för pipeline/analys
OUTPUT_FORMAT = "csv"   # "csv" | "npz"
OUTPUT_PATH   = Path(r"G:\My Drive\Project SHM\Data\mega_dick.csv")    # None => auto: samma namn, nytt suffix

# Sensor-setup
NUM_SENSORS = 2          # 2 nu, 4 senare
AXES_PER_SENSOR = 3      # x,y,z
TOTAL_A = 6             # a0..a11 i din frame (4 sensorer * 3 axlar)

# Counts -> g (BYT till korrekt scale factor för din konfig)
LSB_PER_G = 256000.0

# CSV options
CSV_INCLUDE_RAW = True   # inkludera a0..a11 i output-CSV

# Metadata (spårbarhet)
WRITE_META_JSON = True
META_PATH = Path(r"decoded\decode_meta.json")  # sätt None om du vill stänga av

# =======================
# Frame-layout: samma som din UDP-logger :contentReference[oaicite:0]{index=0}
# little-endian: magic u32, seq u32, t_us u32, a[12] i32
STRUCT_FMT = "<III" + ("i" * TOTAL_A)
FRAME_SIZE = struct.calcsize(STRUCT_FMT)
MAGIC_OK = 0xA55A5AA5


def read_frames_from_bin(path: Path):
    """
    Läser rå BIN som består av frames hopskrivna efter varandra.
    Resyncar genom att scanna efter MAGIC (robust om filen inte är perfekt alignad).
    """
    data = path.read_bytes()
    n = len(data)

    seqs, t_uses, a_list = [], [], []
    i = 0

    while i + FRAME_SIZE <= n:
        magic = int.from_bytes(data[i:i+4], "little", signed=False)
        if magic != MAGIC_OK:
            i += 1
            continue

        chunk = data[i:i+FRAME_SIZE]
        magic_u, seq, t_us, *a = struct.unpack(STRUCT_FMT, chunk)
        if magic_u != MAGIC_OK:
            i += 1
            continue

        seqs.append(seq)
        t_uses.append(t_us)
        a_list.append(a)
        i += FRAME_SIZE

    if not seqs:
        raise ValueError(
            f"Inga frames hittades i {path}. "
            f"Kontrollera att filen innehåller råa frames (FRAME_SIZE={FRAME_SIZE})."
        )

    return {
        "seq": np.array(seqs, dtype=np.uint32),
        "t_us": np.array(t_uses, dtype=np.uint32),
        "a": np.array(a_list, dtype=np.int32),  # shape (N, 12)
    }


def read_frames_from_csv(path: Path):
    """
    Läser CSV med header:
    recv_time_s,seq,t_us,a0..a11
    Tomma a-fält -> NaN
    """
    recv, seqs, t_uses, a_list = [], [], [], []

    with path.open(newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            recv.append(float(row["recv_time_s"]))
            seqs.append(int(row["seq"]) & 0xFFFFFFFF)
            t_uses.append(int(row["t_us"]) & 0xFFFFFFFF)

            a_row = []
            for i in range(TOTAL_A):
                v = row.get(f"a{i}", "")
                if v is None or str(v).strip() == "":
                    a_row.append(np.nan)
                else:
                    a_row.append(int(v))
            a_list.append(a_row)

    if not seqs:
        raise ValueError(f"Inga rader hittades i {path}.")

    return {
        "recv_time_s": np.array(recv, dtype=np.float64),
        "seq": np.array(seqs, dtype=np.uint32),
        "t_us": np.array(t_uses, dtype=np.uint32),
        "a": np.array(a_list, dtype=np.float64),  # float pga NaN
    }


def apply_counts_to_g_all(frames, lsb_per_g: float, num_sensors: int, axes_per_sensor: int):
    """
    Mappar:
      sensor 0: a0,a1,a2
      sensor 1: a3,a4,a5
      sensor 2: a6,a7,a8
      sensor 3: a9,a10,a11
    Skapar:
      s0_ax_g..s{N-1}_az_g
    """
    a = frames["a"].astype(np.float64)  # int32 -> float64, behåller ev NaN

    max_sensors_possible = a.shape[1] // axes_per_sensor
    if num_sensors > max_sensors_possible:
        raise ValueError(
            f"NUM_SENSORS={num_sensors} men datan har bara plats för {max_sensors_possible} sensorer "
            f"({a.shape[1]} värden / {axes_per_sensor} axlar)."
        )

    for s in range(num_sensors):
        base = s * axes_per_sensor
        frames[f"s{s}_ax_g"] = a[:, base + 0] / lsb_per_g
        frames[f"s{s}_ay_g"] = a[:, base + 1] / lsb_per_g
        frames[f"s{s}_az_g"] = a[:, base + 2] / lsb_per_g

    return frames


def write_csv(frames, out_path: Path, num_sensors: int, include_raw: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = []
    if "recv_time_s" in frames:
        header.append("recv_time_s")
    header += ["seq", "t_us"]

    # g-kolumner per sensor
    for s in range(num_sensors):
        header += [f"s{s}_ax_g", f"s{s}_ay_g", f"s{s}_az_g"]

    if include_raw:
        header += [f"a{i}" for i in range(TOTAL_A)]

    N = len(frames["seq"])
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i in range(N):
            row = []
            if "recv_time_s" in frames:
                row.append(f"{frames['recv_time_s'][i]:.6f}")

            row += [int(frames["seq"][i]), int(frames["t_us"][i])]

            # g per sensor
            for s in range(num_sensors):
                for comp in ("ax", "ay", "az"):
                    v = frames[f"s{s}_{comp}_g"][i]
                    row.append("" if (isinstance(v, float) and not np.isfinite(v)) else f"{v:.8f}")

            # rå counts
            if include_raw:
                for v in frames["a"][i, :]:
                    if isinstance(v, float) and not np.isfinite(v):
                        row.append("")
                    else:
                        row.append(int(v))

            w.writerow(row)


def write_npz(frames, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **frames)


def main():
    # --- input format ---
    in_fmt = INPUT_FORMAT.lower()
    if in_fmt == "auto":
        suf = INPUT_PATH.suffix.lower()
        if suf == ".csv":
            in_fmt = "csv"
        elif suf == ".bin":
            in_fmt = "bin"
        else:
            raise ValueError("INPUT_FORMAT=auto men okänt suffix. Sätt INPUT_FORMAT till 'bin' eller 'csv'.")

    # --- load ---
    if in_fmt == "bin":
        frames = read_frames_from_bin(INPUT_PATH)
    else:
        frames = read_frames_from_csv(INPUT_PATH)

    # --- convert ---
    frames = apply_counts_to_g_all(frames, LSB_PER_G, NUM_SENSORS, AXES_PER_SENSOR)

    # --- output path ---
    if OUTPUT_PATH is None:
        out_ext = ".csv" if OUTPUT_FORMAT == "csv" else ".npz"
        out_path = INPUT_PATH.with_suffix(out_ext)
    else:
        out_path = Path(OUTPUT_PATH)

    # --- write ---
    if OUTPUT_FORMAT == "csv":
        write_csv(frames, out_path, NUM_SENSORS, CSV_INCLUDE_RAW)
    elif OUTPUT_FORMAT == "npz":
        write_npz(frames, out_path)
    else:
        raise ValueError("OUTPUT_FORMAT måste vara 'csv' eller 'npz'.")

    # --- meta ---
    if WRITE_META_JSON and META_PATH is not None:
        meta = {
            "input_path": str(INPUT_PATH),
            "input_format": in_fmt,
            "output_path": str(out_path),
            "output_format": OUTPUT_FORMAT,
            "num_sensors": NUM_SENSORS,
            "axes_per_sensor": AXES_PER_SENSOR,
            "total_a": TOTAL_A,
            "lsb_per_g": LSB_PER_G,
            "frame_size": FRAME_SIZE,
            "magic_ok": f"0x{MAGIC_OK:08X}",
            "csv_include_raw": CSV_INCLUDE_RAW,
        }
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"OK: {INPUT_PATH} ({in_fmt}) -> {out_path} ({OUTPUT_FORMAT})")


if __name__ == "__main__":
    main()
