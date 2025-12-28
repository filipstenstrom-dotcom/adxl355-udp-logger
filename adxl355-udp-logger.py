#!/usr/bin/env python3
import socket
import struct
import time
import csv
from pathlib import Path

# ======= KONFIG =======
LISTEN_IP   = "0.0.0.0"
LISTEN_PORT = 5000

# STM32 frame_t layout (little-endian):
# magic u32, seq u32, t_us u32, a[12] i32
STRUCT_FMT = "<III" + ("i" * 12)
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
    print(f"Writing: {bin_path} and {csv_path}")

    last_seq = None
    total = 0
    dropped = 0

    with open(bin_path, "wb") as fbin, open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        header = ["recv_time_s", "seq", "t_us"] + [f"a{i}" for i in range(12)]
        writer.writerow(header)

        t0 = time.time()

        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                # ingen data just nu, fortsätt
                continue

            # Spara rått binärt exakt som det kom (alltid bra att ha)
            fbin.write(data)

            # Tolka endast paket som matchar exakt frame-storleken
            if len(data) != FRAME_SIZE:
                print(f"Skip: packet size {len(data)} from {addr}, expected {FRAME_SIZE}")
                continue

            magic, seq, t_us, *a = struct.unpack(STRUCT_FMT, data)
            if magic != MAGIC_OK:
                print(f"Skip: bad magic 0x{magic:08X} from {addr}")
                continue

            # räkna tappade paket (UDP kan tappa)
            if last_seq is not None and seq != (last_seq + 1) & 0xFFFFFFFF:
                # konservativt: räkna som drop om hopp framåt
                if seq > last_seq:
                    dropped += (seq - last_seq - 1)
                else:
                    # wrap-around eller omstart; räkna inte aggressivt
                    pass
            last_seq = seq

            now = time.time() - t0
            writer.writerow([f"{now:.6f}", seq, t_us] + a)

            total += 1
            if total % 1000 == 0:
                print(f"Frames: {total}, dropped(est): {dropped}, last seq: {last_seq}")

if __name__ == "__main__":
    main()