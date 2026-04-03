#!/usr/bin/env python3
import socket
import struct
import time
import csv
from pathlib import Path

# ======= KONFIG =======
LISTEN_IP   = "0.0.0.0"
LISTEN_PORT = 5000

# frame_t:
# u32 magic, u32 seq, u32 t_us, i32 a[12], i16 temp[4]
STRUCT_FMT = "<III" + ("i" * 12) + ("h" * 4)
FRAME_SIZE = struct.calcsize(STRUCT_FMT)
MAGIC_OK   = 0xA55A5AA5

OUT_DIR = Path("udp_capture")
OUT_DIR.mkdir(exist_ok=True)
# =======================

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    sock.settimeout(1.0)

    ts = time.strftime("%Y%m%d_%H%M%S")
    bin_path = OUT_DIR / f"frames_{ts}.bin"
    csv_path = OUT_DIR / f"frames_{ts}.csv"

    print(f"Listening UDP on {LISTEN_IP}:{LISTEN_PORT}")
    print(f"Expecting frame size: {FRAME_SIZE} bytes")

    last_seq = None
    total = 0
    dropped = 0

    with open(bin_path, "wb") as fbin, open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)

        header = (
            ["recv_time_s", "seq", "t_us"]
            + [f"a{i}" for i in range(12)]
            + [f"temp{i}_C" for i in range(4)]
        )
        writer.writerow(header)

        t0 = time.time()

        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue

            fbin.write(data)

            if len(data) < FRAME_SIZE:
                continue

            unpacked = struct.unpack(STRUCT_FMT, data[:FRAME_SIZE])

            magic = unpacked[0]
            seq   = unpacked[1]
            t_us  = unpacked[2]
            a     = list(unpacked[3:15])
            temp_raw = list(unpacked[15:19])

            if magic != MAGIC_OK:
                continue

            # konvertera centi-degC -> °C
            temp_C = [tr / 100.0 for tr in temp_raw]

            if last_seq is not None and seq != (last_seq + 1) & 0xFFFFFFFF:
                if seq > last_seq:
                    dropped += (seq - last_seq - 1)
            last_seq = seq

            now = time.time() - t0
            writer.writerow(
                [f"{now:.6f}", seq, t_us] +
                a +
                [f"{t:.2f}" for t in temp_C]
            )

            total += 1
            if total % 1000 == 0:
                print(f"Frames: {total}, dropped(est): {dropped}, last seq: {last_seq}")

if __name__ == "__main__":
    main()