"""
fog_collector_wifi.py
=====================
Data collector for URIC Wearable System.

Setup:
    1. Flash fog_data_logger_wifi.ino to ESP32
    2. Turn on phone hotspot: ysfthecreator / ysfthecreator@123
    3. Connect laptop to same hotspot
    4. Check ESP32 OLED for IP address
    5. Run: python fog_collector_wifi.py --ip <IP_FROM_OLED>

Labels:
    0=Stand  1=Walk  2=FastWalk  3=Sit  4=SitStand
    5=Stairs  6=Fall  7=FoG  8=Seizure
"""

import socket
import os
import csv
import threading
import argparse
from datetime import datetime

ESP32_PORT = 8888
BUFFER     = 4096

LABEL_NAMES = {
    0: "Stand",
    1: "Walk",
    2: "FastWalk",
    3: "Sit",
    4: "SitStand",
    5: "Stairs",
    6: "Fall",
    7: "FoG",
    8: "Seizure"
}

RAW_COLS = ["timestamp_ms","ax","ay","az","gx","gy","gz","label","label_name"]

def make_dirs():
    os.makedirs("data/raw", exist_ok=True)

def ts_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def open_session(ts, label):
    rp = f"data/raw/session_{ts}_{LABEL_NAMES[label]}.csv"
    rf = open(rp, "w", newline="")
    rw = csv.writer(rf)
    rw.writerow(RAW_COLS)
    return rf, rw, rp

input_queue = []
input_lock  = threading.Lock()

def input_thread_fn():
    while True:
        try:
            line = input()
            with input_lock:
                input_queue.append(line.strip())
        except EOFError:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True, help="ESP32 IP from OLED display")
    args = parser.parse_args()

    make_dirs()

    print("\n" + "="*55)
    print("  URIC Wearable System - Data Collector")
    print("="*55)
    print("\n  LABELS:")
    for k, v in LABEL_NAMES.items():
        print(f"    {k} = {v}")
    print("\n  CONTROLS:")
    print("    0-8  → set label")
    print("    S    → start / stop recording")
    print("    C    → calibrate (keep still 10s)")
    print("    R    → reset counter")
    print("    Q    → quit")
    print("="*55 + "\n")

    print(f"🔌 Connecting to {args.ip}:{ESP32_PORT}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((args.ip, ESP32_PORT))
        sock.settimeout(0.1)
        print("✅ Connected!\n")
    except Exception as e:
        print(f"❌ Failed: {e}")
        print(f"   Run: python fog_collector_wifi.py --ip <IP>")
        return

    threading.Thread(target=input_thread_fn, daemon=True).start()

    current_label = 0
    ts = ts_str()
    rf, rw, rp = open_session(ts, current_label)
    raw_count  = 0
    in_session = False
    remainder  = ""

    try:
        while True:
            with input_lock:
                cmds = input_queue[:]
                input_queue.clear()

            for cmd in cmds:
                cmd = cmd.strip()
                upper = cmd.upper()
                if upper == "Q":
                    raise KeyboardInterrupt
                elif upper == "S":
                    sock.sendall(b"S\n")
                elif upper == "C":
                    sock.sendall(b"C\n")
                    print("  → Calibrating... keep device flat and still!")
                elif upper == "R":
                    sock.sendall(b"R\n")
                elif cmd in [str(i) for i in range(9)]:
                    sock.sendall(cmd.encode())
                    current_label = int(cmd)
                    print(f"\n  → Label: {current_label} = {LABEL_NAMES[current_label]}")

            try:
                data = sock.recv(BUFFER).decode("utf-8", errors="ignore")
                if data:
                    remainder += data
            except socket.timeout:
                pass
            except Exception as e:
                print(f"\n❌ Socket error: {e}")
                break

            while "\n" in remainder:
                line, remainder = remainder.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                if "SESSION START" in line:
                    in_session = True
                    ts = ts_str()
                    rf.close()
                    rf, rw, rp = open_session(ts, current_label)
                    raw_count = 0
                    print(f"\n📹 Recording → {rp}")
                    continue

                if "SESSION END" in line:
                    in_session = False
                    duration = raw_count // 50
                    print(f"\n🛑 Done! {raw_count} samples ({duration}s)")
                    print(f"   Saved: {rp}\n")
                    continue

                if line.startswith("#"):
                    msg = line[2:].strip()
                    if msg and "FEAT" not in msg:
                        print(f"  [{msg}]")
                    continue

                if in_session:
                    parts = line.split(",")
                    if len(parts) == len(RAW_COLS):
                        rw.writerow(parts)
                        raw_count += 1
                        if raw_count % 100 == 0:
                            duration = raw_count // 50
                            print(f"  ⏺  {raw_count} samples ({duration}s) | {LABEL_NAMES[current_label]}      ", end="\r")

    except KeyboardInterrupt:
        print("\n\n👋 Stopped.")
    finally:
        try:
            sock.sendall(b"S\n")
            sock.close()
        except:
            pass
        rf.close()
        print(f"\nFinal: {rp} ({raw_count} samples = {raw_count//50}s)")

if __name__ == "__main__":
    main()
