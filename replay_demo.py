# -*- coding: utf-8 -*-
"""
sahwa Demo Replay Script
Simulates ESP32 streaming by replaying recorded CSV data over TCP.
Run this AFTER starting sahwa_server.py
"""

import socket
import time
import os
import sys
import csv
import random

# ============================================================
# CONFIG
# ============================================================
SERVER_IP   = "127.0.0.1"
SERVER_PORT = 8888
SAMPLE_RATE = 50          # Hz — matches training
INTERVAL    = 1.0 / SAMPLE_RATE

# Data folder — adjust if needed
DATA_DIR = os.path.join("data", "sahwa_data", "wrist")

# Scenario sequence — cycles through these automatically
# Format: (csv_filename, mode_int, display_label, duration_seconds)
SCENARIOS = [
    ("Stand.csv",    0, "STANDING STILL",     20),
    ("Walk.csv",     0, "NORMAL WALKING",      25),
    ("FastWalk.csv", 0, "FAST WALKING",        20),
    ("Sit.csv",      0, "SITTING",             20),
    ("Walk.csv",     0, "WALKING AGAIN",       15),
    ("Fall.csv",     0, "SIMULATED FALL",      20),
    ("Walk.csv",     0, "RECOVERY WALKING",    15),
    ("Seizure.csv",  0, "SIMULATED SEIZURE",   25),
    ("Stand.csv",    0, "POST-EVENT STANDING", 20),
    ("Walk.csv",     0, "NORMAL WALKING",      20),
]

# ============================================================
# HELPERS
# ============================================================
def load_csv(filepath):
    rows = []
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "ax": float(row["ax"]),
                        "ay": float(row["ay"]),
                        "az": float(row["az"]),
                        "gx": float(row["gx"]),
                        "gy": float(row["gy"]),
                        "gz": float(row["gz"]),
                    })
                except (KeyError, ValueError):
                    continue
    except FileNotFoundError:
        print("[Replay] WARNING: File not found: " + filepath)
    return rows


def format_line(row, ts_ms):
    return (str(ts_ms) + "," +
            "{:.4f}".format(row["ax"]) + "," +
            "{:.4f}".format(row["ay"]) + "," +
            "{:.4f}".format(row["az"]) + "," +
            "{:.4f}".format(row["gx"]) + "," +
            "{:.4f}".format(row["gy"]) + "," +
            "{:.4f}".format(row["gz"]) + "\n")


def add_noise(row, scale=0.005):
    """Add tiny realistic noise to avoid perfectly identical windows."""
    return {k: v + random.gauss(0, scale) for k, v in row.items()}


def connect(ip, port, retries=10):
    for attempt in range(1, retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            print("[Replay] Connected to server at " + ip + ":" + str(port))
            return sock
        except ConnectionRefusedError:
            print("[Replay] Attempt " + str(attempt) + "/" + str(retries) +
                  " — server not ready, retrying in 2s...")
            time.sleep(2)
    print("[Replay] Could not connect. Is sahwa_server.py running?")
    sys.exit(1)


def send_mode(sock, mode):
    try:
        sock.sendall(("MODE:" + str(mode) + "\n").encode("utf-8"))
    except Exception as e:
        raise ConnectionError("Send failed: " + str(e))


def stream_scenario(sock, rows, duration_sec, label):
    if not rows:
        print("[Replay] Skipping empty dataset: " + label)
        time.sleep(duration_sec)
        return

    samples_needed = int(duration_sec * SAMPLE_RATE)
    ts_ms          = 0
    idx            = 0

    print("[Replay] Streaming: " + label +
          " (" + str(duration_sec) + "s, " + str(samples_needed) + " samples)")

    start = time.perf_counter()
    for i in range(samples_needed):
        row  = add_noise(rows[idx % len(rows)])
        idx += 1
        line = format_line(row, ts_ms).encode("utf-8")
        try:
            sock.sendall(line)
        except Exception as e:
            raise ConnectionError("Stream broken: " + str(e))

        ts_ms += int(INTERVAL * 1000)

        # Precise 50Hz timing
        target = start + (i + 1) * INTERVAL
        wait   = target - time.perf_counter()
        if wait > 0:
            time.sleep(wait)

    elapsed = time.perf_counter() - start
    print("[Replay] Done: " + label +
          " — {:.1f}s elapsed".format(elapsed))

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  sahwa Demo Replay v1.0")
    print("  Simulates ESP32 streaming for dashboard demo")
    print("=" * 60)
    print("[Replay] Data directory: " + os.path.abspath(DATA_DIR))
    print("[Replay] Connecting to: " + SERVER_IP + ":" + str(SERVER_PORT))
    print()

    # Pre-load all CSV files
    data_cache = {}
    for filename, _, label, _ in SCENARIOS:
        if filename not in data_cache:
            path = os.path.join(DATA_DIR, filename)
            rows = load_csv(path)
            data_cache[filename] = rows
            status = str(len(rows)) + " rows" if rows else "NOT FOUND"
            print("[Replay] Loaded " + filename + ": " + status)

    print()
    print("[Replay] Scenario sequence:")
    for i, (fname, mode, label, dur) in enumerate(SCENARIOS):
        mname = "Wrist" if mode == 0 else "Ankle"
        print("  " + str(i+1).zfill(2) + ". " + label.ljust(28) + str(dur) + "s  [" + mname + "]")
    print()

    sock = connect(SERVER_IP, SERVER_PORT)

    print("[Replay] Starting in 3 seconds...")
    time.sleep(3)

    scenario_num = 0
    try:
        while True:
            scenario_num += 1
            cycle = ((scenario_num - 1) % len(SCENARIOS))
            filename, mode, label, duration = SCENARIOS[cycle]

            print()
            print("[Replay] === Scenario " + str(scenario_num) +
                  " / Cycle " + str(cycle + 1) + "/" + str(len(SCENARIOS)) +
                  ": " + label + " ===")

            send_mode(sock, mode)
            rows = data_cache.get(filename, [])
            stream_scenario(sock, rows, duration, label)

            # Brief pause between scenarios
            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print("[Replay] Stopped by user.")
    except ConnectionError as e:
        print("[Replay] Connection lost: " + str(e))
        print("[Replay] Restart the server and run replay_demo.py again.")
    finally:
        try:
            sock.close()
        except Exception:
            pass
        print("[Replay] Done.")


if __name__ == "__main__":
    main()
