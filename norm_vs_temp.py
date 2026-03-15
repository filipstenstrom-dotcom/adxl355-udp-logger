#!/usr/bin/env python3
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =======================
# SETTINGS
# =======================

INPUT_GLOB = r"G:/My Drive/Project SHM/Data/Temp_verification/Calibrated_temp/*.csv"
OUT_DIR = Path("G:/My Drive/Project SHM/Data/Info/Temp")

OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SENSORS = 2   # ändra till 4 senare

# Välj vilken typ av filer du analyserar:
# 1) rå counts a0..a11
USE_RAW_COUNTS = False
LSB_PER_G = 256000.0

# 2) redan kalibrerade kolumner s0_ax_g, ...
USE_CALIBRATED_COLUMNS = True

# 3) eller raw_g-kolumner s0_ax_g_raw, ...
USE_RAW_G_COLUMNS = False

START_OFFSET_S = 1.0
TEMP_BIN_WIDTH_C = 1.0
USE_MEDIAN_PER_BIN = False

TEMP_COL_TEMPLATE = "temp{sensor}_C"

# Outlier removal
REMOVE_OUTLIERS = True
OUTLIER_STD_MULT = 3.0   # t.ex. 2.5 eller 3.0

# =======================
# Helpers
# =======================
def get_axis_columns(sensor_index: int):
    if USE_RAW_COUNTS:
        base = sensor_index * 3
        return (f"a{base+0}", f"a{base+1}", f"a{base+2}")

    if USE_CALIBRATED_COLUMNS:
        return (
            f"s{sensor_index}_ax_g",
            f"s{sensor_index}_ay_g",
            f"s{sensor_index}_az_g",
        )

    if USE_RAW_G_COLUMNS:
        return (
            f"s{sensor_index}_ax_g_raw",
            f"s{sensor_index}_ay_g_raw",
            f"s{sensor_index}_az_g_raw",
        )

    raise ValueError("Ogiltig kolumnkonfiguration.")

def temp_bin_center(temp_c: float, bin_width: float):
    return np.round(temp_c / bin_width) * bin_width

def process_one_sensor_in_file(df: pd.DataFrame, sensor_index: int, path: Path):
    temp_col = TEMP_COL_TEMPLATE.format(sensor=sensor_index)
    ax_col, ay_col, az_col = get_axis_columns(sensor_index)

    required = ["recv_time_s", temp_col, ax_col, ay_col, az_col]
    for c in required:
        if c not in df.columns:
            return None

    t = df["recv_time_s"].to_numpy(dtype=float)
    mask_time = np.isfinite(t) & (t >= (np.nanmin(t) + START_OFFSET_S))
    if np.count_nonzero(mask_time) < 10:
        return None

    temp_arr = df[temp_col].to_numpy(dtype=float)

    if USE_RAW_COUNTS:
        ax = df[ax_col].to_numpy(dtype=float) / LSB_PER_G
        ay = df[ay_col].to_numpy(dtype=float) / LSB_PER_G
        az = df[az_col].to_numpy(dtype=float) / LSB_PER_G
    else:
        ax = df[ax_col].to_numpy(dtype=float)
        ay = df[ay_col].to_numpy(dtype=float)
        az = df[az_col].to_numpy(dtype=float)

    mask = (
        mask_time
        & np.isfinite(ax)
        & np.isfinite(ay)
        & np.isfinite(az)
        & np.isfinite(temp_arr)
    )
    if np.count_nonzero(mask) < 10:
        return None

    ax_sel = ax[mask]
    ay_sel = ay[mask]
    az_sel = az[mask]
    temp_sel = temp_arr[mask]

    mx = float(np.mean(ax_sel))
    my = float(np.mean(ay_sel))
    mz = float(np.mean(az_sel))

    norm_g = float(np.sqrt(mx * mx + my * my + mz * mz))
    norm_err_g = norm_g - 1.0

    temp_mean = float(np.mean(temp_sel))
    tbin = float(temp_bin_center(temp_mean, TEMP_BIN_WIDTH_C))

    return {
        "file": path.name,
        "sensor": sensor_index,
        "temp_C": temp_mean,
        "temp_bin_C": tbin,
        "mx_g": mx,
        "my_g": my,
        "mz_g": mz,
        "norm_g": norm_g,
        "norm_err_g": norm_err_g,
        "sample_count": int(np.count_nonzero(mask)),
    }

def remove_outliers_sigma(df: pd.DataFrame, value_col: str, std_mult: float):
    """
    Tar bort outliers globalt per sensor baserat på mean/std.
    """
    if len(df) < 3:
        df = df.copy()
        df["is_outlier"] = False
        return df

    mu = df[value_col].mean()
    sigma = df[value_col].std(ddof=1)

    if not np.isfinite(sigma) or sigma == 0:
        df = df.copy()
        df["is_outlier"] = False
        return df

    df = df.copy()
    df["is_outlier"] = np.abs(df[value_col] - mu) > (std_mult * sigma)
    return df

def summarize_sensor(per_file_sensor: pd.DataFrame):
    if USE_MEDIAN_PER_BIN:
        grouped = (
            per_file_sensor.groupby("temp_bin_C")
            .agg(
                temp_mean_C=("temp_C", "median"),
                norm_mean_g=("norm_g", "median"),
                norm_err_mean_g=("norm_err_g", "median"),
                mx_mean_g=("mx_g", "median"),
                my_mean_g=("my_g", "median"),
                mz_mean_g=("mz_g", "median"),
                count=("norm_g", "size"),
            )
            .reset_index()
        )
        std_df = (
            per_file_sensor.groupby("temp_bin_C")
            .agg(
                norm_std_g=("norm_g", "std"),
                norm_err_std_g=("norm_err_g", "std"),
            )
            .reset_index()
        )
        summary = grouped.merge(std_df, on="temp_bin_C", how="left")
    else:
        summary = (
            per_file_sensor.groupby("temp_bin_C")
            .agg(
                temp_mean_C=("temp_C", "mean"),
                norm_mean_g=("norm_g", "mean"),
                norm_std_g=("norm_g", "std"),
                norm_err_mean_g=("norm_err_g", "mean"),
                norm_err_std_g=("norm_err_g", "std"),
                mx_mean_g=("mx_g", "mean"),
                my_mean_g=("my_g", "mean"),
                mz_mean_g=("mz_g", "mean"),
                count=("norm_g", "size"),
            )
            .reset_index()
        )
    return summary.sort_values("temp_bin_C").reset_index(drop=True)

def plot_sensor(per_file_before: pd.DataFrame,
                per_file_after: pd.DataFrame,
                summary: pd.DataFrame,
                sensor_index: int):

    # Plot 1: norm error vs temp
    plt.figure(figsize=(10, 6))

    plt.scatter(
        per_file_before["temp_C"],
        per_file_before["norm_err_g"],
        alpha=0.35,
        label="Per file (before outlier removal)"
    )

    if len(per_file_after) > 0:
        plt.scatter(
            per_file_after["temp_C"],
            per_file_after["norm_err_g"],
            alpha=0.8,
            label="Per file (used)"
        )

    plt.errorbar(
        summary["temp_bin_C"],
        summary["norm_err_mean_g"],
        yerr=summary["norm_err_std_g"].fillna(0.0),
        fmt="o-",
        capsize=4,
        label=f"Binned mean ± std ({TEMP_BIN_WIDTH_C:.1f} °C)"
    )

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Norm error [g]  (|mean(a)| - 1)")
    plt.title(f"Sensor {sensor_index}: Norm error vs temperature")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"sensor{sensor_index}_norm_error_vs_temp.png", dpi=180)
    plt.close()

    # Plot 2: norm vs temp
    plt.figure(figsize=(10, 6))

    plt.scatter(
        per_file_before["temp_C"],
        per_file_before["norm_g"],
        alpha=0.35,
        label="Per file (before outlier removal)"
    )

    if len(per_file_after) > 0:
        plt.scatter(
            per_file_after["temp_C"],
            per_file_after["norm_g"],
            alpha=0.8,
            label="Per file (used)"
        )

    plt.errorbar(
        summary["temp_bin_C"],
        summary["norm_mean_g"],
        yerr=summary["norm_std_g"].fillna(0.0),
        fmt="o-",
        capsize=4,
        label=f"Binned mean ± std ({TEMP_BIN_WIDTH_C:.1f} °C)"
    )

    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Norm [g]  (|mean(a)|)")
    plt.title(f"Sensor {sensor_index}: Norm vs temperature")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"sensor{sensor_index}_norm_vs_temp.png", dpi=180)
    plt.close()

# =======================
# Main
# =======================
def main():
    files = [Path(p) for p in sorted(glob.glob(INPUT_GLOB))]
    if not files:
        raise ValueError("Hittade inga CSV-filer. Kontrollera INPUT_GLOB.")

    all_rows = []

    for path in files:
        try:
            df = pd.read_csv(path)
            for sensor_index in range(NUM_SENSORS):
                row = process_one_sensor_in_file(df, sensor_index, path)
                if row is not None:
                    all_rows.append(row)
        except Exception as e:
            print(f"Skip {path.name}: {e}")

    if not all_rows:
        raise ValueError("Inga giltiga data hittades.")

    per_file = pd.DataFrame(all_rows).sort_values(["sensor", "temp_C", "file"]).reset_index(drop=True)
    per_file.to_csv(OUT_DIR / "all_sensors_per_file_before_outliers.csv", index=False)

    all_summaries = []

    for sensor_index in range(NUM_SENSORS):
        per_file_sensor = per_file[per_file["sensor"] == sensor_index].copy()
        if per_file_sensor.empty:
            print(f"Sensor {sensor_index}: inga giltiga filer.")
            continue

        # Outlier removal
        if REMOVE_OUTLIERS:
            tagged = remove_outliers_sigma(
                per_file_sensor,
                value_col="norm_err_g",
                std_mult=OUTLIER_STD_MULT
            )
        else:
            tagged = per_file_sensor.copy()
            tagged["is_outlier"] = False

        outliers = tagged[tagged["is_outlier"]].copy()
        used = tagged[~tagged["is_outlier"]].copy()

        tagged.to_csv(OUT_DIR / f"sensor{sensor_index}_per_file_with_outlier_flag.csv", index=False)
        outliers.to_csv(OUT_DIR / f"sensor{sensor_index}_outliers.csv", index=False)
        used.to_csv(OUT_DIR / f"sensor{sensor_index}_per_file_used.csv", index=False)

        summary = summarize_sensor(used)
        summary["sensor"] = sensor_index
        all_summaries.append(summary)

        summary.to_csv(OUT_DIR / f"sensor{sensor_index}_summary.csv", index=False)

        plot_sensor(per_file_sensor, used, summary, sensor_index)

        print(f"Sensor {sensor_index}: total={len(per_file_sensor)}, used={len(used)}, outliers={len(outliers)}")

    if all_summaries:
        summary_all = pd.concat(all_summaries, ignore_index=True)
        summary_all.to_csv(OUT_DIR / "all_sensors_summary.csv", index=False)

    print("Done.")
    print(f"Results in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()