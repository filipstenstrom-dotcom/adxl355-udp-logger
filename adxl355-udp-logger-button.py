#!/usr/bin/env python3
import socket
import struct
import time
import csv
from pathlib import Path

# ======= KONFIG =======
LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5000

# Vanlig sensorframe:
# u32 magic, u32 seq, u32 t_us, i32 a[12], i16 temp[4]
FRAME_FMT = "<III" + ("i" * 12) + ("h" * 4)
FRAME_SIZE = struct.calcsize(FRAME_FMT)
MAGIC_FRAME = 0xA55A5AA5

# START/STOP event:
# u32 magic, u32 seq, u32 t_us, u32 capture_id, u32 sample_count
EVT_FMT = "<IIIII"
EVT_SIZE = struct.calcsize(EVT_FMT)
MAGIC_START = 0x53544152  # 'STAR'
MAGIC_STOP  = 0x53544F50  # 'STOP'

OUT_DIR = Path("udp_capture")
OUT_DIR.mkdir(exist_ok=True)
# =======================


def open_capture_files(capture_id: int):
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"frames_{ts}_cap{capture_id:04d}"
    bin_path = OUT_DIR / f"{base}.bin"
    csv_path = OUT_DIR / f"{base}.csv"

    fbin = open(bin_path, "wb")
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(fcsv)

    header = (
        ["recv_time_s", "seq", "t_us"]
        + [f"a{i}" for i in range(12)]
        + [f"temp{i}_C" for i in range(4)]
    )
    writer.writerow(header)

    return {
        "bin_path": bin_path,
        "csv_path": csv_path,
        "fbin": fbin,
        "fcsv": fcsv,
        "writer": writer,
        "t0": time.time(),
        "frame_count": 0,
        "dropped": 0,
        "last_seq": None,
    }


def close_capture_files(cap):
    if cap is None:
        return
    try:
        cap["fcsv"].flush()
        cap["fcsv"].close()
    finally:
        cap["fbin"].flush()
        cap["fbin"].close()


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    sock.settimeout(1.0)

    print(f"Listening UDP on {LISTEN_IP}:{LISTEN_PORT}")
    print(f"Expecting frame size: {FRAME_SIZE} bytes")
    print(f"Expecting event size: {EVT_SIZE} bytes")

    current_capture = None
    current_capture_id = None

    while True:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue

        if len(data) < 4:
            continue

        magic = struct.unpack_from("<I", data, 0)[0]

        # =========================
        # START / STOP event packet
        # =========================
        if magic in (MAGIC_START, MAGIC_STOP):
            if len(data) < EVT_SIZE:
                print(f"[WARN] Event packet too short: {len(data)} bytes")
                continue

            evt_magic, evt_seq, evt_t_us, capture_id, sample_count = struct.unpack(EVT_FMT, data[:EVT_SIZE])

            if evt_magic == MAGIC_START:
                # Om något redan är öppet, stäng det först
                if current_capture is not None:
                    print(f"[WARN] START received while capture {current_capture_id} still open. Closing previous capture.")
                    close_capture_files(current_capture)

                current_capture = open_capture_files(capture_id)
                current_capture_id = capture_id

                print(
                    f"[START] capture_id={capture_id} "
                    f"sample_count={sample_count} "
                    f"seq={evt_seq} t_us={evt_t_us} "
                    f"-> {current_capture['csv_path'].name}"
                )
                continue

            if evt_magic == MAGIC_STOP:
                if current_capture is None:
                    print(f"[WARN] STOP received for capture_id={capture_id}, but no capture is open.")
                    continue

                print(
                    f"[STOP] capture_id={capture_id} "
                    f"seq={evt_seq} t_us={evt_t_us} "
                    f"frames_written={current_capture['frame_count']} "
                    f"dropped_est={current_capture['dropped']}"
                )
                close_capture_files(current_capture)
                current_capture = None
                current_capture_id = None
                continue

        # =========================
        # Normal sensor frame
        # =========================
        if magic == MAGIC_FRAME:
            if len(data) < FRAME_SIZE:
                print(f"[WARN] Data frame too short: {len(data)} bytes")
                continue

            # Skriv bara om capture är aktiv
            if current_capture is None:
                continue

            unpacked = struct.unpack(FRAME_FMT, data[:FRAME_SIZE])

            _magic = unpacked[0]
            seq = unpacked[1]
            t_us = unpacked[2]
            a = list(unpacked[3:15])
            temp_raw = list(unpacked[15:19])

            # extra säkerhet
            if _magic != MAGIC_FRAME:
                continue

            current_capture["fbin"].write(data[:FRAME_SIZE])

            # konvertera centi-degC -> °C
            temp_C = [tr / 100.0 for tr in temp_raw]

            # enkel packet loss-estimat
            last_seq = current_capture["last_seq"]
            if last_seq is not None:
                expected = (last_seq + 1) & 0xFFFFFFFF
                if seq != expected:
                    if seq > last_seq:
                        current_capture["dropped"] += (seq - last_seq - 1)

            current_capture["last_seq"] = seq

            now = time.time() - current_capture["t0"]
            current_capture["writer"].writerow(
                [f"{now:.6f}", seq, t_us]
                + a
                + [f"{t:.2f}" for t in temp_C]
            )

            current_capture["frame_count"] += 1

            if current_capture["frame_count"] % 1000 == 0:
                print(
                    f"[CAP {current_capture_id}] "
                    f"Frames: {current_capture['frame_count']}, "
                    f"dropped(est): {current_capture['dropped']}, "
                    f"last seq: {current_capture['last_seq']}"
                )

            continue

        # =========================
        # Okänd packet-typ
        # =========================
        print(f"[WARN] Unknown magic: 0x{magic:08X}, len={len(data)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")