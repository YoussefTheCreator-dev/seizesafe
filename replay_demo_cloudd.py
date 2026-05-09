# -*- coding: utf-8 -*-
"""
sahwa Cloud Replay Demo
Streams recorded CSV data to the Railway server via WebSocket.
Works with the cloud deployment at sahwa-monitor.up.railway.app

Install deps first:
    pip install python-socketio[client] websocket-client

Usage:
    python replay_demo_cloud.py
    python replay_demo_cloud.py --url https://sahwa-monitor.up.railway.app
    python replay_demo_cloud.py --url http://localhost:5000   (for local testing)
"""

import sys
import os
import csv
import time
import random
import argparse
import threading

# ── Try importing socketio ────────────────────────────────────────────
try:
    import socketio
except ImportError:
    print("[ERROR] Missing dependency. Run:")
    print("   pip install 'python-socketio[client]' websocket-client")
    sys.exit(1)

# ── CONFIG ────────────────────────────────────────────────────────────
DEFAULT_URL  = "https://sahwa-monitor.up.railway.app"
DATA_DIR     = os.path.join("sahwa_data", "wrist")
SAMPLE_RATE  = 50
INTERVAL     = 1.0 / SAMPLE_RATE

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

# ── HELPERS ───────────────────────────────────────────────────────────
def load_csv(filepath):
    rows = []
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "ax": float(row["ax"]), "ay": float(row["ay"]), "az": float(row["az"]),
                        "gx": float(row["gx"]), "gy": float(row["gy"]), "gz": float(row["gz"]),
                    })
                except (KeyError, ValueError):
                    continue
    except FileNotFoundError:
        print("[Replay] WARNING: File not found: " + filepath)
    return rows

def add_noise(row, scale=0.004):
    return {k: v + random.gauss(0, scale) for k, v in row.items()}

def fmt_line(row, ts_ms):
    return (str(ts_ms) + "," +
            "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                row["ax"], row["ay"], row["az"],
                row["gx"], row["gy"], row["gz"]))

# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="sahwa Cloud Replay")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help="Server URL (default: %(default)s)")
    args = parser.parse_args()

    url = args.url.rstrip("/")

    print("=" * 58)
    print("  sahwa Cloud Replay")
    print("  Streaming to: " + url)
    print("=" * 58)

    # Pre-load all CSV files
    data_cache = {}
    for filename, _, label, _ in SCENARIOS:
        if filename not in data_cache:
            path = os.path.join(DATA_DIR, filename)
            rows = load_csv(path)
            data_cache[filename] = rows
            status = str(len(rows)) + " rows" if rows else "NOT FOUND"
            print("[Data] " + filename + ": " + status)

    missing = [f for f, r in data_cache.items() if not r]
    if missing:
        print("\n[ERROR] Missing data files: " + str(missing))
        print("[ERROR] Make sure sahwa_data/wrist/ folder is present")
        sys.exit(1)

    print("\n[Scenarios]")
    for i, (f, m, label, dur) in enumerate(SCENARIOS):
        print("  {:02d}. {:<28} {}s".format(i+1, label, dur))

    # Create socket.io client — use polling (works without websocket-client)
    sio = socketio.Client(
        reconnection=True,
        reconnection_attempts=10,
        reconnection_delay=2,
        logger=False,
        engineio_logger=False,
    )

    connected = threading.Event()

    @sio.event
    def connect():
        print("\n[WS] Connected to server!")
        connected.set()

    @sio.event
    def disconnect():
        print("[WS] Disconnected from server")
        connected.clear()

    @sio.on("show_notification")
    def on_notif(data):
        print("[Server] " + str(data.get("message", "")))

    @sio.on("critical_alert")
    def on_alert(data):
        print("[ALERT] " + data.get("type","?") + " at " + data.get("time","?"))

    # Connect — try websocket first, fall back to polling
    print("\n[WS] Connecting to " + url + " ...")
    connected_ok = False
    for transport in [["websocket", "polling"], ["polling"]]:
        try:
            sio.connect(
                url,
                namespaces=["/esp32"],
                transports=transport,
                wait_timeout=10,
            )
            connected_ok = True
            break
        except Exception as e:
            print("[WS] Transport {} failed: {}".format(transport[0], str(e)[:80]))
            try: sio.disconnect()
            except: pass

    if not connected_ok:
        print("\n[ERROR] Could not connect. Trying without namespace (direct emit)...")
        try:
            sio.connect(url, transports=["polling"], wait_timeout=10)
            connected_ok = True
            print("[WS] Connected without namespace — will emit to default namespace")
        except Exception as e:
            print("[WS] Final attempt failed: " + str(e)[:120])
            print("\nMake sure Railway server is deployed and running.")
            sys.exit(1)

    print("[WS] Connected! Starting in 3 seconds...")
    time.sleep(3)

    scenario_num = 0
    try:
        while True:
            cycle = scenario_num % len(SCENARIOS)
            filename, mode, label, duration = SCENARIOS[cycle]
            scenario_num += 1

            print("\n[Replay] === Scenario {:02d} / Cycle {:02d}/10: {} ===".format(
                scenario_num, cycle+1, label))

            # Send mode switch
            try:
                try:
                    sio.emit("mode_change", {"mode": mode}, namespace="/esp32")
                except Exception:
                    sio.emit("mode_change", {"mode": mode})
            except Exception as e:
                print("[WS] Mode emit error: " + str(e))
                break

            rows = data_cache.get(filename, [])
            if not rows:
                print("[Replay] Skipping empty: " + label)
                time.sleep(duration)
                continue

            samples_needed = int(duration * SAMPLE_RATE)
            ts_ms = 0
            idx   = 0

            print("[Replay] Streaming {} ({} samples)...".format(label, samples_needed))
            start = time.perf_counter()

            for i in range(samples_needed):
                if not sio.connected:
                    print("[WS] Lost connection during stream")
                    raise ConnectionError("Disconnected")

                row  = add_noise(rows[idx % len(rows)])
                idx += 1
                line = fmt_line(row, ts_ms)

                try:
                    try:
                        sio.emit("imu_line", line, namespace="/esp32")
                    except Exception:
                        sio.emit("imu_line", line)
                except Exception as e:
                    print("[WS] Emit error: " + str(e))
                    raise ConnectionError(str(e))

                ts_ms += int(INTERVAL * 1000)

                # Maintain 50 Hz pacing
                target = start + (i + 1) * INTERVAL
                wait   = target - time.perf_counter()
                if wait > 0.001:
                    time.sleep(wait)

            elapsed = time.perf_counter() - start
            print("[Replay] Done: {} — {:.1f}s elapsed".format(label, elapsed))
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Replay] Stopped by user.")
    except ConnectionError as e:
        print("[Replay] Connection lost: " + str(e))
    finally:
        try:
            sio.disconnect()
        except Exception:
            pass
        print("[Replay] Done.")


if __name__ == "__main__":
    main()
