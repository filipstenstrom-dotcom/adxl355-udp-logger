#!/usr/bin/env python3
import socket
import struct
import time
import csv
from pathlib import Path

# ======= KONFIG =======
LISTEN_IP   = "0.0.0.0"
LISTEN_PORT = 5000

# adxl355_id_sanity_pkt_t layout (little-endian)
# uint32 magic
# uint32 seq
# uint32 t_us
# uint8  sensor_count
# uint8  reserved[3]
# then sensor_count * (4 bytes: devid_ad, devid_mst, partid, revid)

MAGIC_OK = 0x53444941  # 'AIDS'
HEADER_FMT = "<III B 3s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
ID_FMT = "<BBBB"
ID_SIZE = struct.calcsize(ID_FMT)

OUT_DIR = Path("udp_id_sanity")
OUT_DIR.mkdir(exist_ok=True)
# =======================


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, LISTEN_PORT))
    sock.settimeout(1.0)

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"id_sanity_{ts}.csv"

    print(f"Listening UDP on {LISTEN_IP}:{LISTEN_PORT}")
    print(f"Writing CSV: {csv_path}")

    last_ids = {}   # sensor_index -> (devid_ad, devid_mst, partid, revid)
    last_seq = None

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "recv_time_s",
            "seq",
            "t_us",
            "sensor",
            "devid_ad",
            "devid_mst",
            "partid",
            "revid",
            "changed"
        ])

        t0 = time.time()

        while True:
            try:
                data, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue

            if len(data) < HEADER_SIZE:
                print(f"Skip: packet too small ({len(data)})")
                continue

            magic, seq, t_us, sensor_count, _ = struct.unpack(
                HEADER_FMT, data[:HEADER_SIZE]
            )

            if magic != MAGIC_OK:
                print(f"Bad magic 0x{magic:08X} (len={len(data)})")
                continue

            # Sekvenskontroll (inte kritisk, men bra info)
            if last_seq is not None and seq != ((last_seq + 1) & 0xFFFFFFFF):
                print(f"Seq jump: {last_seq} -> {seq}")
            last_seq = seq

            offset = HEADER_SIZE
            now = time.time() - t0

            for i in range(sensor_count):
                if offset + ID_SIZE > len(data):
                    break

                ids = struct.unpack(ID_FMT, data[offset:offset + ID_SIZE])
                offset += ID_SIZE

                prev = last_ids.get(i)
                changed = (prev is not None and prev != ids)
                last_ids[i] = ids

                writer.writerow([
                    f"{now:.6f}",
                    seq,
                    t_us,
                    i,
                    ids[0],  # devid_ad
                    ids[1],  # devid_mst
                    ids[2],  # partid
                    ids[3],  # revid
                    int(changed)
                ])
                fcsv.flush()

                if changed:
                    print(
                        f"ID CHANGE sensor {i}! "
                        f"prev={prev}, now={ids}"
                    )


if __name__ == "__main__":
    main()
