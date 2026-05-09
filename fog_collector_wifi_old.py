"""
fog_collector_wifi.py
=====================
Connects to ESP32-C3 over WiFi TCP and saves labeled IMU data.

Setup:
    1. Flash fog_data_logger_wifi.ino to your ESP32
    2. Turn on your phone hotspot: ysfthecreator / ysfthecreator@123
    3. Connect your LAPTOP to the same hotspot
    4. Check ESP32 OLED for its IP address
    5. Run: python fog_collector_wifi.py --ip <IP_FROM_OLED>

Output:
    data/raw/session_YYYYMMDD_HHMMSS.csv       <- raw IMU samples
    data/features/session_YYYYMMDD_HHMMSS.csv  <- windowed features
"""

import socket
import os
import csv
import sys
import threading
import argparse
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
ESP32_PORT = 8888
BUFFER     = 4096

# ── Updated Labels ────────────────────────────────────────────────────────────
# 0=Stand  1=Walk  2=FoG  3=Fall  4=Seizure
LABEL_NAMES = {0:"Stand", 1:"Walk", 2:"FoG", 3:"Fall", 4:"Seizure"}

# ── Column definitions ────────────────────────────────────────────────────────
RAW_COLS  = ["timestamp_ms","ax","ay","az","gx","gy","gz","label","label_name"]
FEAT_COLS = ["timestamp_ms","rms_acc","rms_gyro","freeze_index",
             "mean_acc","std_acc","energy_acc","zero_crossings","peak_acc","label"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_dirs():
    os.makedirs("data/raw",      exist_ok=True)
    os.makedirs("data/features", exist_ok=True)

def ts_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def open_session(ts):
    rp = f"data/raw/session_{ts}.csv"
    fp = f"data/features/session_{ts}.csv"
    rf = open(rp, "w", newline="")
    ff = open(fp, "w", newline="")
    rw = csv.writer(rf)
    fw = csv.writer(ff)
    rw.writerow(RAW_COLS)
    fw.writerow(FEAT_COLS)
    return rf, ff, rw, fw, rp, fp

# ── Keyboard input thread ─────────────────────────────────────────────────────
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

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="URIC Wearable Data Collector")
    parser.add_argument("--ip", required=True, help="ESP32 IP shown on OLED display")
    args = parser.parse_args()

    make_dirs()

    print("\n" + "="*55)
    print("  URIC Wearable System - Data Collector")
    print("="*55)
    print("\n⚡ Checklist:")
    print("   1. Phone hotspot ON (ysfthecreator)")
    print("   2. ESP32 powered and connected to hotspot")
    print("   3. Laptop connected to same hotspot\n")

    print(f"🔌 Connecting to ESP32 at {args.ip}:{ESP32_PORT}...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((args.ip, ESP32_PORT))
        sock.settimeout(0.1)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"\n   Check IP on OLED and retry:")
        print(f"   python fog_collector_wifi.py --ip <IP>")
        return

    print("✅ Connected!\n")
    print("─"*55)
    print("  LABELS:")
    for k, v in LABEL_NAMES.items():
        print(f"    {k} = {v}")
    print("─"*55)
    print("  CONTROLS (type + Enter):")
    print("    0-4  → set label")
    print("    S    → start / stop recording")
    print("    C    → calibrate (keep still 10s)")
    print("    R    → reset")
    print("    Q    → quit")
    print("─"*55 + "\n")

    t = threading.Thread(target=input_thread_fn, daemon=True)
    t.start()

    ts = ts_str()
    rf, ff, rw, fw, rp, fp = open_session(ts)

    raw_count    = 0
    feat_count   = 0
    in_session   = False
    current_label = 1
    remainder    = ""

    try:
        while True:
            # Handle keyboard input
            with input_lock:
                cmds = input_queue[:]
                input_queue.clear()

            for cmd in cmds:
                cmd = cmd.upper().strip()
                if cmd == "Q":
                    raise KeyboardInterrupt
                elif cmd == "S":
                    sock.sendall(b"S\n")
                elif cmd in ("0","1","2","3","4"):
                    sock.sendall(cmd.encode())
                    current_label = int(cmd)
                    print(f"\n  → Label: {LABEL_NAMES[current_label]}")
                elif cmd == "C":
                    sock.sendall(b"C\n")
                    print("  → Calibrating... keep still")
                elif cmd == "R":
                    sock.sendall(b"R\n")
                    print("  → Reset")

            # Read from socket
            try:
                data = sock.recv(BUFFER).decode("utf-8", errors="ignore")
                if data:
                    remainder += data
            except socket.timeout:
                pass
            except Exception as e:
                print(f"\n❌ Socket error: {e}")
                break

            # Process lines
            while "\n" in remainder:
                line, remainder = remainder.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                if "SESSION START" in line:
                    in_session = True
                    ts = ts_str()
                    rf.close(); ff.close()
                    rf, ff, rw, fw, rp, fp = open_session(ts)
                    raw_count = 0; feat_count = 0
                    print(f"\n📹 Recording → {rp}")
                    print(f"   Label: {LABEL_NAMES[current_label]}")
                    continue

                if "SESSION END" in line:
                    in_session = False
                    duration = raw_count // 50
                    print(f"\n🛑 Stopped. {raw_count} samples ({duration}s)")
                    print(f"   Raw:      {rp}")
                    print(f"   Features: {fp}\n")
                    continue

                if line.startswith("# FEAT,"):
                    parts = line[7:].split(",")
                    if len(parts) == len(FEAT_COLS):
                        fw.writerow(parts)
                        feat_count += 1
                    continue

                if line.startswith("#"):
                    msg = line[2:].strip()
                    if msg:
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
        ff.close()
        print(f"\nSaved:")
        print(f"  {rp}  ({raw_count} samples = {raw_count//50}s)")
        print(f"  {fp}  ({feat_count} windows)")

if __name__ == "__main__":
    main()
