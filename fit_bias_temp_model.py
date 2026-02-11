#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import csv

# =======================
# SETTINGS
# =======================

INPUT_PATH = Path(r"G:\My Drive\Project SHM\Data\frames_with_temp.csv")  # .csv eller .npz
NUM_SENSORS = 2

# Vilken kolumn innehåller temperatur?
# I CSV: namn på kolumn. I NPZ: array key.
TEMP_COLUMN = "temp_C"

# Tid
TIME_COLUMN = "recv_time_s"
FS_HZ = None  # om du saknar tid, ange t.ex. 4000.0

# Fönster för bias-estimat (sekunder)
WINDOW_S = 60.0     # 30–120s brukar vara bra
STEP_S   = 60.0     # samma som WINDOW för icke-overlap

# LPF för bias-estimat (Hz)
# Vi gör en enkel 1:a ordningens IIR på sample-basis för enkelhetens skull.
# Om din FS är hög och du vill exakt, kan vi byta till Butterworth offline.
LPF_CUTOFF_HZ = 0.5

# Orientering under hela temp-run:
# vilken axel bär gravitation och vilket tecken?
GRAV_AXIS = "z"   # "x"|"y"|"z"
GRAV_SIGN = +1    # +1 eller -1

# Output
OUTPUT_JSON = Path(r"decoded\bias_temp_model.json")

def colnames(s):
    return (f"s{s}_ax_g", f"s{s}_ay_g", f"s{s}_az_g")

def safe_float(v):
    if v is None: return np.nan
    s = str(v).strip()
    if s == "": return np.nan
    return float(s)

def load_csv(path: Path):
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise ValueError("CSV tom.")

    cols = rows[0].keys()

    # tid
    if TIME_COLUMN in cols:
        t = np.array([float(row[TIME_COLUMN]) for row in rows], dtype=np.float64)
    else:
        if FS_HZ is None:
            raise ValueError(f"Saknar {TIME_COLUMN} och FS_HZ=None.")
        t = np.arange(len(rows), dtype=np.float64) / float(FS_HZ)

    # temp
    if TEMP_COLUMN not in cols:
        raise ValueError(f"Saknar TEMP_COLUMN={TEMP_COLUMN} i CSV.")
    temp = np.array([safe_float(row.get(TEMP_COLUMN, "")) for row in rows], dtype=np.float64)

    data = {"t": t, "temp_C": temp}

    for s in range(NUM_SENSORS):
        cx, cy, cz = colnames(s)
        for c in (cx, cy, cz):
            if c not in cols:
                raise ValueError(f"Saknar kolumn {c}.")
        data[cx] = np.array([safe_float(row.get(cx, "")) for row in rows], dtype=np.float64)
        data[cy] = np.array([safe_float(row.get(cy, "")) for row in rows], dtype=np.float64)
        data[cz] = np.array([safe_float(row.get(cz, "")) for row in rows], dtype=np.float64)

    return data

def load_npz(path: Path):
    z = np.load(path, allow_pickle=False)
    keys = set(z.keys())

    # tid
    if TIME_COLUMN in keys:
        t = z[TIME_COLUMN].astype(np.float64)
    else:
        if FS_HZ is None:
            raise ValueError(f"Saknar {TIME_COLUMN} i NPZ och FS_HZ=None.")
        n = len(z["seq"]) if "seq" in keys else len(next(iter(z.values())))
        t = np.arange(n, dtype=np.float64) / float(FS_HZ)

    if TEMP_COLUMN not in keys:
        raise ValueError(f"Saknar TEMP_COLUMN={TEMP_COLUMN} i NPZ.")
    temp = z[TEMP_COLUMN].astype(np.float64)

    data = {"t": t, "temp_C": temp}

    for s in range(NUM_SENSORS):
        cx, cy, cz = colnames(s)
        for c in (cx, cy, cz):
            if c not in keys:
                raise ValueError(f"Saknar {c} i NPZ.")
        data[cx] = z[cx].astype(np.float64)
        data[cy] = z[cy].astype(np.float64)
        data[cz] = z[cz].astype(np.float64)

    return data

def lpf_iir(x, fs, fc):
    # enkel 1:a ordningens lågpass: y[n]=y[n-1]+a*(x[n]-y[n-1])
    # a = 1 - exp(-2*pi*fc/fs)
    a = 1.0 - np.exp(-2.0*np.pi*fc/fs)
    y = np.empty_like(x, dtype=np.float64)
    y0 = x[0]
    y[0] = y0
    for i in range(1, len(x)):
        xi = x[i]
        y0 = y0 + a*(xi - y0)
        y[i] = y0
    return y

def main():
    if INPUT_PATH.suffix.lower() == ".csv":
        data = load_csv(INPUT_PATH)
    elif INPUT_PATH.suffix.lower() == ".npz":
        data = load_npz(INPUT_PATH)
    else:
        raise ValueError("INPUT_PATH måste vara .csv eller .npz")

    t = data["t"]
    temp = data["temp_C"]

    # uppskatta fs från tidskolumn
    if len(t) < 2:
        raise ValueError("För få samples.")
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Ogiltig tidskolumn.")
    fs = 1.0 / dt

    # förväntad gravitation i denna fasta orientering
    expected = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    expected[{"x":0,"y":1,"z":2}[GRAV_AXIS]] = float(GRAV_SIGN) * 1.0

    # LPF på axlarna (för bias-estimat)
    lpf = {}
    for s in range(NUM_SENSORS):
        cx, cy, cz = colnames(s)
        lpf[cx] = lpf_iir(data[cx], fs, LPF_CUTOFF_HZ)
        lpf[cy] = lpf_iir(data[cy], fs, LPF_CUTOFF_HZ)
        lpf[cz] = lpf_iir(data[cz], fs, LPF_CUTOFF_HZ)

    # windowing för bias-punkter
    t_start = float(t[0])
    t_end = float(t[-1])

    points = {str(s): {"T": [], "bx": [], "by": [], "bz": []} for s in range(NUM_SENSORS)}

    w0 = t_start
    while w0 + WINDOW_S <= t_end:
        w1 = w0 + WINDOW_S
        mask = (t >= w0) & (t <= w1) & np.isfinite(temp)
        if np.count_nonzero(mask) < 10:
            w0 += STEP_S
            continue

        Tm = float(np.nanmean(temp[mask]))

        for s in range(NUM_SENSORS):
            cx, cy, cz = colnames(s)
            mx = float(np.nanmean(lpf[cx][mask]))
            my = float(np.nanmean(lpf[cy][mask]))
            mz = float(np.nanmean(lpf[cz][mask]))

            m = np.array([mx, my, mz], dtype=np.float64)
            b = m - expected  # biaspunkt

            points[str(s)]["T"].append(Tm)
            points[str(s)]["bx"].append(float(b[0]))
            points[str(s)]["by"].append(float(b[1]))
            points[str(s)]["bz"].append(float(b[2]))

        w0 += STEP_S

    # Fit linjär modell per axel: b(T) = b0 + k*(T - T0)
    out = {
        "input_path": str(INPUT_PATH),
        "method": "bias(T) linear fit from still windows (LPF for estimation only)",
        "LPF_CUTOFF_HZ": LPF_CUTOFF_HZ,
        "WINDOW_S": WINDOW_S,
        "STEP_S": STEP_S,
        "orientation": {"grav_axis": GRAV_AXIS, "grav_sign": GRAV_SIGN},
        "fs_estimated_hz": float(fs),
        "sensors": {}
    }

    for s in range(NUM_SENSORS):
        S = str(s)
        T = np.array(points[S]["T"], dtype=np.float64)
        bx = np.array(points[S]["bx"], dtype=np.float64)
        by = np.array(points[S]["by"], dtype=np.float64)
        bz = np.array(points[S]["bz"], dtype=np.float64)

        if T.size < 5:
            raise ValueError(f"För få bias-punkter för sensor {s} (fick {T.size}). Behöver längre logg eller fler fönster.")

        T0 = float(np.nanmean(T))

        def fit_line(y):
            x = T - T0
            A = np.vstack([np.ones_like(x), x]).T
            # least squares
            p, *_ = np.linalg.lstsq(A, y, rcond=None)
            b0 = float(p[0])
            k = float(p[1])
            # residual std
            yhat = A @ p
            resid = y - yhat
            rstd = float(np.nanstd(resid, ddof=2))
            return b0, k, rstd

        b0x, kx, rx = fit_line(bx)
        b0y, ky, ry = fit_line(by)
        b0z, kz, rz = fit_line(bz)

        out["sensors"][S] = {
            "T0_C": T0,
            "bias0_g": [b0x, b0y, b0z],
            "k_g_per_C": [kx, ky, kz],
            "resid_std_g": [rx, ry, rz],
            "points": {
                "count": int(T.size),
                "T_min": float(np.min(T)),
                "T_max": float(np.max(T)),
            }
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"OK: wrote {OUTPUT_JSON}")
    for s, d in out["sensors"].items():
        print(f"Sensor {s}: T0={d['T0_C']:.2f}C  bias0_g={np.array(d['bias0_g'])}  k(g/C)={np.array(d['k_g_per_C'])}")

if __name__ == "__main__":
    main()
