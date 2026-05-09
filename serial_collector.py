"""
serial_collector.py
===================
Collects IMU data from ESP32 via USB Serial cable.
Restored for original FastIMU firmware format.
"""

import serial
import serial.tools.list_ports
import os
import csv
import threading
import argparse
import time
from datetime import datetime

BAUD_RATE  = 115200
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

def list_ports():
    ports = serial.tools.list_ports.comports()
    if ports:
        print("\n  Available COM ports:")
        for p in ports:
            print(f"    {p.device} - {p.description}")
    else:
        print("\n  No COM ports found. Is the ESP32 connected?")

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
    parser.add_argument("--port", help="COM port e.g. COM10")
    parser.add_argument("--list", action="store_true", help="List available ports")
    args = parser.parse_args()

    if args.list or not args.port:
        list_ports()
        if not args.port:
            print("\n  Run: python serial_collector.py --port COM10")
            return

    make_dirs()

    print("\n" + "="*55)
    print("  URIC Wearable System - USB Serial Collector (Restored)")
    print("="*55)
    print("\n  LABELS:")
    for k, v in LABEL_NAMES.items():
        print(f"    {k} = {v}")
    print("\n  CONTROLS (type + Enter):")
    print("    0-8          -> set label")
    print("    S            -> start / stop recording")
    print("    start <0-8>  -> set label and start")
    print("    stop         -> stop recording")
    print("    Q            -> quit")
    print("="*55 + "\n")

    print(f"Opening {args.port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(args.port, BAUD_RATE, timeout=0.1)
        ser.dtr = True
        ser.rts = True
        print("Connected via USB!\n")
    except Exception as e:
        print(f"Failed: {e}")
        list_ports()
        return

    threading.Thread(target=input_thread_fn, daemon=True).start()

    current_label = 0
    ts = ts_str()
    # Dummy session
    rf, rw, rp = open_session(ts, current_label)
    raw_count  = 0
    in_session = False
    remainder  = ""

    try:
        while True:
            # Handle keyboard input
            with input_lock:
                cmds = input_queue[:]
                input_queue.clear()

            for cmd in cmds:
                cmd = cmd.strip()
                upper = cmd.upper()
                if upper == "Q":
                    raise KeyboardInterrupt
                elif upper == "S" or upper == "STOP":
                    ser.write(b"S\n")
                elif upper.startswith("START"):
                    parts = cmd.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        label = parts[1]
                        ser.write(label.encode())
                        current_label = int(label)
                        print(f"\n  -> Label: {current_label} = {LABEL_NAMES[current_label]}")
                        time.sleep(0.5)
                        ser.write(b"S\n")
                    else:
                        print("  -> Invalid start command. Use 'start <0-8>'")
                elif cmd in [str(i) for i in range(9)]:
                    ser.write(cmd.encode())
                    current_label = int(cmd)
                    print(f"\n  -> Label: {current_label} = {LABEL_NAMES[current_label]}")

            # Read from serial
            try:
                data = ser.read(BUFFER).decode("utf-8", errors="ignore")
                if data:
                    remainder += data
            except Exception as e:
                print(f"\nSerial error: {e}")
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
                    rf.close()
                    rf, rw, rp = open_session(ts, current_label)
                    raw_count = 0
                    print(f"\nRecording -> {rp}")
                    print(f"   Label: {LABEL_NAMES[current_label]}")
                    continue

                if "SESSION END" in line:
                    in_session = False
                    duration = raw_count // 50
                    mins = duration // 60
                    secs = duration % 60
                    print(f"\nStopped. {raw_count} samples ({mins}m {secs}s)")
                    print(f"   Saved: {rp}\n")
                    continue

                if in_session:
                    parts = line.split(",")
                    if len(parts) == len(RAW_COLS):
                        rw.writerow(parts)
                        raw_count += 1
                        if raw_count % 100 == 0:
                            duration = raw_count // 50
                            mins = duration // 60
                            secs = duration % 60
                            print(f"  *  {raw_count} samples ({mins}m {secs}s) | {LABEL_NAMES[current_label]}      ", end="\r")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        try:
            ser.write(b"S\n")
            ser.close()
        except:
            pass
        rf.close()
        duration = raw_count // 50
        print(f"\nFinal: {rp} ({raw_count} samples = {duration//60}m {duration%60}s)")

if __name__ == "__main__":
    main()
