# -*- coding: utf-8 -*-
"""
sahwa Server v1.0
Real-Time Neurological Event Monitoring System
ESP32-C3 + MPU6500 -> TCP -> ML Inference -> Web Dashboard
"""

import os
import sys
import json
import socket
import threading
import time
import webbrowser
import smtplib
import io
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import joblib
from flask import Flask, render_template_string, make_response, jsonify
from flask_socketio import SocketIO

# ============================================================
# CONFIGURATION - Edit these values
# ============================================================
TCP_HOST = "0.0.0.0"
TCP_PORT = 8888
WEB_PORT = 5000

PATIENT_NAME = "Patient"
CAREGIVER_EMAIL = "caregiver@example.com"
GMAIL_SENDER = "sahwa.alerts@gmail.com"
GMAIL_APP_PASSWORD = "your-app-password"

WINDOW_SIZE = 256
SAMPLE_RATE = 50
EPISODES_FILE = "episodes.json"

WRIST_LABELS = {0: "Stand", 1: "Walk", 2: "FastWalk", 3: "Sit", 4: "SitStand", 6: "Fall", 8: "Seizure"}
ANKLE_LABELS = {0: "Stand", 1: "Walk", 2: "FastWalk", 3: "Sit", 4: "SitStand", 5: "Stairs", 7: "FoG"}
CRITICAL_WRIST = {6, 8}
CRITICAL_ANKLE = {7}
WARNING_LABELS = {"FOG", "SITSTAND"}

# ============================================================
# SHARED STATE
# ============================================================
data_lock = threading.Lock()
imu_buffer = deque()
esp_socket_ref = [None]

state = {
    "device_mode": 0,
    "esp_connected": False,
    "prediction": "Waiting for device...",
    "pred_label": -1,
    "color": "#00aaff",
    "is_critical": False,
    "samples_received": 0,
    "last_update": "",
}

episode_log = []
current_episode = None
normal_streak = 0

# ============================================================
# FLASK & SOCKETIO
# ============================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = "sahwa2025"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ============================================================
# FEATURE EXTRACTION (must match training exactly)
# ============================================================
def spectral_entropy(psd):
    psd = psd[psd > 0]
    if len(psd) == 0:
        return 0.0
    psd = psd / psd.sum()
    return float(-np.sum(psd * np.log2(psd)) / np.log2(len(psd)))


def extract_features(df):
    df = df.copy()
    df["accel_magnitude"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["gyro_magnitude"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    signals = ["ax", "ay", "az", "gx", "gy", "gz", "accel_magnitude", "gyro_magnitude"]
    feats = []
    for sig in signals:
        d = df[sig].values.astype(float)
        feats.append(float(np.mean(d)))
        feats.append(float(np.std(d)))
        feats.append(float(np.min(d)))
        feats.append(float(np.max(d)))
        feats.append(float(np.sqrt(np.mean(d**2))))
        centered = d - np.mean(d)
        zcr = np.where(np.diff(np.sign(centered)))[0]
        feats.append(float(len(zcr) / len(d)))
        n = len(d)
        yf = fft(d)
        mags = np.abs(yf[1:n // 2])
        freqs = fftfreq(n, 1.0 / SAMPLE_RATE)[1:n // 2]
        if len(mags) > 0:
            feats.append(float(freqs[np.argmax(mags)]))
            feats.append(spectral_entropy(mags**2))
        else:
            feats.extend([0.0, 0.0])
    return np.array(feats).reshape(1, -1)

# ============================================================
# MODEL LOADER
# ============================================================
class ModelPack:
    def __init__(self, model_path, scaler_path, mapping_path, critical_set, label_names):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(mapping_path, "r") as f:
            raw = json.load(f)
        self.mapping = {int(k): v for k, v in raw.items()}
        self.critical = critical_set
        self.label_names = label_names

    def predict(self, df):
        feats = extract_features(df)
        scaled = self.scaler.transform(feats)
        pred_idx = int(self.model.predict(scaled)[0])
        orig_label = next((k for k, v in self.mapping.items() if v == pred_idx), -1)
        name = self.label_names.get(orig_label, "Unknown")
        is_crit = orig_label in self.critical
        return orig_label, name, is_crit


def load_models():
    models = {}
    wrist_files = ["wrist_xgb_model.pkl", "wrist_xgb_scaler.pkl", "wrist_label_mapping.json"]
    ankle_files = ["ankle_rf_model.pkl", "ankle_rf_scaler.pkl", "ankle_label_mapping.json"]
    if all(os.path.exists(f) for f in wrist_files):
        models[0] = ModelPack("wrist_xgb_model.pkl", "wrist_xgb_scaler.pkl",
                               "wrist_label_mapping.json", CRITICAL_WRIST, WRIST_LABELS)
        print("[Models] Wrist model loaded (XGBoost)")
    else:
        print("[Models] WARNING: Wrist model files not found. Wrist mode disabled.")
    if all(os.path.exists(f) for f in ankle_files):
        models[1] = ModelPack("ankle_rf_model.pkl", "ankle_rf_scaler.pkl",
                               "ankle_label_mapping.json", CRITICAL_ANKLE, ANKLE_LABELS)
        print("[Models] Ankle model loaded (Random Forest)")
    else:
        print("[Models] WARNING: Ankle model files not found. Ankle mode disabled.")
    if not models:
        print("[Models] ERROR: No model files found. Exiting.")
        sys.exit(1)
    return models

# ============================================================
# EPISODE MANAGER
# ============================================================
def start_episode(event_type):
    global current_episode
    current_episode = {
        "type": event_type,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        "severity": "CRITICAL" if event_type in ["SEIZURE", "FALL"] else "WARNING",
    }


def end_episode():
    global current_episode, episode_log
    if current_episode:
        current_episode["end_time"] = datetime.now().isoformat()
        start = datetime.fromisoformat(current_episode["start_time"])
        end = datetime.fromisoformat(current_episode["end_time"])
        current_episode["duration_seconds"] = int((end - start).total_seconds())
        episode_log.append(current_episode)
        save_episodes()
        socketio.emit("update_log", {"log": get_log_json()})
        socketio.emit("update_stats", get_stats())
        current_episode = None


def save_episodes():
    with open(EPISODES_FILE, "w") as f:
        json.dump(episode_log, f, indent=2)


def get_log_json():
    result = []
    for ep in episode_log[-20:]:
        result.append({
            "start_time": ep["start_time"][:19].replace("T", " "),
            "type": ep["type"],
            "duration_seconds": ep["duration_seconds"],
            "severity": ep["severity"],
        })
    return result


def get_stats():
    if not episode_log:
        return {"total": 0, "longest": 0, "most_frequent": "N/A"}
    types = [ep["type"] for ep in episode_log]
    most_freq = max(set(types), key=types.count)
    longest = max(ep["duration_seconds"] for ep in episode_log)
    return {"total": len(episode_log), "longest": longest, "most_frequent": most_freq}

# ============================================================
# ALERT SYSTEM
# ============================================================
def trigger_alert(event_type):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ALERT] {event_type} detected at {now}")
    socketio.emit("critical_alert", {"type": event_type, "time": now})
    send_buzzer_alert()
    send_email_alert(event_type, now)


def send_buzzer_alert():
    with data_lock:
        sock = esp_socket_ref[0]
    if sock:
        try:
            sock.sendall(b"ALERT\n")
        except Exception:
            pass


def send_email_alert(event_type, timestamp):
    if GMAIL_APP_PASSWORD == "your-app-password":
        print("[Email] Skipping - no Gmail app password configured")
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_SENDER
        msg["To"] = CAREGIVER_EMAIL
        msg["Subject"] = "sahwa ALERT: " + event_type + " detected"
        body = (
            "sahwa Emergency Alert\n\n"
            "Patient: " + PATIENT_NAME + "\n"
            "Event: " + event_type + "\n"
            "Time: " + timestamp + "\n\n"
            "Please check on your patient immediately.\n\n"
            "-- sahwa Monitoring System"
        )
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER, CAREGIVER_EMAIL, msg.as_string())
        print("[Email] Alert sent to " + CAREGIVER_EMAIL)
    except Exception as e:
        print("[Email] Failed: " + str(e))

# ============================================================
# INFERENCE ENGINE THREAD
# ============================================================
class InferenceEngine(threading.Thread):
    def __init__(self, models):
        super().__init__(daemon=True)
        self.models = models
        self.running = True

    def run(self):
        global normal_streak, current_episode
        print("[Inference] Engine started.")
        while self.running:
            with data_lock:
                buf_len = len(imu_buffer)
            if buf_len >= WINDOW_SIZE:
                with data_lock:
                    window = [imu_buffer.popleft() for _ in range(WINDOW_SIZE)]
                try:
                    df = pd.DataFrame(window)
                    mode = state["device_mode"]
                    pack = self.models.get(mode)
                    if pack is None:
                        time.sleep(0.1)
                        continue
                    orig_label, name, is_crit = pack.predict(df)
                    name_upper = name.upper()
                    if is_crit:
                        color = "#ff2244"
                    elif name_upper in WARNING_LABELS:
                        color = "#ff9900"
                    else:
                        color = "#00cc66"
                    with data_lock:
                        state["prediction"] = name_upper
                        state["pred_label"] = orig_label
                        state["color"] = color
                        state["is_critical"] = is_crit
                        state["last_update"] = datetime.now().strftime("%H:%M:%S")
                    socketio.emit("update_activity", {
                        "activity": name_upper,
                        "color": color,
                        "is_critical": is_crit,
                        "label": orig_label,
                    })
                    if is_crit:
                        normal_streak = 0
                        if current_episode is None:
                            start_episode(name_upper)
                            trigger_alert(name_upper)
                    else:
                        if current_episode is not None:
                            normal_streak += 1
                            if normal_streak >= 3:
                                end_episode()
                                normal_streak = 0
                        else:
                            normal_streak = 0
                except Exception as e:
                    print("[Inference] Error: " + str(e))
            else:
                time.sleep(0.05)

# ============================================================
# TCP SERVER THREAD
# ============================================================
class TCPServer(threading.Thread):
    def __init__(self, models):
        super().__init__(daemon=True)
        self.models = models
        self.running = True

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((TCP_HOST, TCP_PORT))
        srv.listen(1)
        srv.settimeout(1.0)
        print("[TCP] Server listening on port " + str(TCP_PORT))
        while self.running:
            try:
                conn, addr = srv.accept()
                print("[TCP] ESP32 connected from " + str(addr))
                with data_lock:
                    esp_socket_ref[0] = conn
                    state["esp_connected"] = True
                    imu_buffer.clear()
                socketio.emit("update_status", {
                    "esp_connected": True,
                    "device_mode": state["device_mode"],
                    "ip": str(addr[0]),
                })
                socketio.emit("show_notification", {"message": "ESP32 connected from " + str(addr[0])})
                self.handle_client(conn)
                with data_lock:
                    esp_socket_ref[0] = None
                    state["esp_connected"] = False
                print("[TCP] ESP32 disconnected")
                socketio.emit("update_status", {
                    "esp_connected": False,
                    "device_mode": state["device_mode"],
                    "ip": "",
                })
                socketio.emit("show_notification", {"message": "ESP32 disconnected"})
            except socket.timeout:
                continue
            except Exception as e:
                print("[TCP] Error: " + str(e))
                time.sleep(1)

    def handle_client(self, conn):
        buffer = ""
        conn.settimeout(5.0)
        try:
            while True:
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffer += data.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("MODE:"):
                        self.handle_mode(line)
                    elif line == "TEST_ALERT":
                        print("[TCP] Test alert received")
                        trigger_alert("TEST_ALERT")
                    else:
                        self.parse_imu(line)
        except Exception as e:
            print("[TCP] Client handler error: " + str(e))

    def handle_mode(self, line):
        try:
            mode_val = int(line.split(":")[1])
            with data_lock:
                state["device_mode"] = mode_val
            mode_name = "Wrist" if mode_val == 0 else "Ankle"
            print("[TCP] Mode changed to " + mode_name)
            socketio.emit("update_status", {
                "esp_connected": True,
                "device_mode": mode_val,
                "ip": "",
            })
            socketio.emit("show_notification", {"message": "Mode switched to " + mode_name})
        except Exception as e:
            print("[TCP] Mode parse error: " + str(e))

    def parse_imu(self, line):
        parts = line.split(",")
        if len(parts) < 7:
            return
        try:
            row = {
                "ax": float(parts[1]),
                "ay": float(parts[2]),
                "az": float(parts[3]),
                "gx": float(parts[4]),
                "gy": float(parts[5]),
                "gz": float(parts[6]),
            }
            with data_lock:
                imu_buffer.append(row)
                state["samples_received"] += 1
            socketio.emit("update_graphs", row)
        except Exception:
            pass

# ============================================================
# PDF REPORT
# ============================================================
def generate_pdf():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter,
                                topMargin=0.75*inch, bottomMargin=0.75*inch,
                                leftMargin=inch, rightMargin=inch)
        styles = getSampleStyleSheet()
        elements = []

        title_style = styles["Title"]
        elements.append(Paragraph("sahwa Monitoring Report", title_style))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["Normal"]))
        elements.append(Paragraph("Patient: " + PATIENT_NAME, styles["Normal"]))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("Summary Statistics", styles["Heading2"]))
        stats = get_stats()
        summary_data = [
            ["Metric", "Value"],
            ["Total Episodes", str(stats["total"])],
            ["Longest Episode", str(stats["longest"]) + " seconds"],
            ["Most Frequent Event", stats["most_frequent"]],
        ]
        t = Table(summary_data, colWidths=[3*inch, 3*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("Episode Log", styles["Heading2"]))
        if episode_log:
            log_data = [["Time", "Event", "Duration", "Severity"]]
            for ep in reversed(episode_log):
                log_data.append([
                    ep["start_time"][:19].replace("T", " "),
                    ep["type"],
                    str(ep["duration_seconds"]) + "s",
                    ep["severity"],
                ])
            t2 = Table(log_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ]))
            elements.append(t2)
        else:
            elements.append(Paragraph("No episodes recorded.", styles["Normal"]))

        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Recommendations", styles["Heading2"]))
        if stats["total"] == 0:
            rec = "No critical events detected. Continue monitoring."
        elif stats["total"] < 3:
            rec = "A small number of episodes detected. Monitor closely and consult your neurologist."
        else:
            rec = ("Multiple episodes detected (" + str(stats["total"]) + " total). "
                   "Urgent review recommended. Consider medication adjustment.")
        elements.append(Paragraph(rec, styles["Normal"]))
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("-- sahwa Automated Report --", styles["Normal"]))

        doc.build(elements)
        buf.seek(0)
        return buf
    except Exception as e:
        print("[PDF] Error: " + str(e))
        return None

# ============================================================
# DASHBOARD HTML
# ============================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>sahwa Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #040810;
    --surface: #0a1020;
    --surface2: #0f1830;
    --border: #1a2a4a;
    --accent: #00aaff;
    --accent2: #00ffcc;
    --green: #00cc66;
    --orange: #ff9900;
    --red: #ff2244;
    --text: #c8d8f0;
    --text2: #6080a0;
    --mono: 'Share Tech Mono', monospace;
    --sans: 'Rajdhani', sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse at 20% 20%, rgba(0,170,255,0.04) 0%, transparent 60%),
      radial-gradient(ellipse at 80% 80%, rgba(0,255,204,0.03) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
  }

  /* ALERT BANNER */
  #alert-banner {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0;
    background: var(--red);
    color: white;
    text-align: center;
    padding: 14px;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 3px;
    z-index: 1000;
    animation: flashBanner 0.5s infinite alternate;
  }
  @keyframes flashBanner {
    from { opacity: 1; }
    to { opacity: 0.6; }
  }

  /* HEADER */
  header {
    position: relative;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 28px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .logo-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
  }
  .logo-text {
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .logo-sub {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text2);
    letter-spacing: 1px;
  }
  .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 12px;
    background: var(--surface2);
  }
  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--text2);
  }
  .dot.connected { background: var(--green); box-shadow: 0 0 8px var(--green); animation: pulse 1.5s infinite; }
  .dot.wrist-mode { background: #00ccff; box-shadow: 0 0 8px #00ccff; }
  .dot.ankle-mode { background: #cc44ff; box-shadow: 0 0 8px #cc44ff; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  .btn {
    padding: 8px 18px;
    border: none;
    border-radius: 8px;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    letter-spacing: 1px;
    transition: all 0.2s;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--accent), #0066cc);
    color: white;
  }
  .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,170,255,0.4); }
  .btn-outline {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
  }
  .btn-outline:hover { border-color: var(--accent); color: var(--accent); }

  /* MAIN GRID */
  .main {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: 1fr 320px;
    grid-template-rows: auto auto 1fr;
    gap: 16px;
    padding: 20px 28px;
    min-height: calc(100vh - 72px);
  }

  /* ACTIVITY DISPLAY */
  .activity-panel {
    grid-column: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 30px;
    min-height: 200px;
    position: relative;
    overflow: hidden;
  }
  .activity-panel::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,170,255,0.05) 0%, transparent 70%);
    pointer-events: none;
  }
  .activity-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text2);
    letter-spacing: 3px;
    margin-bottom: 12px;
  }
  #activity-display {
    font-family: var(--sans);
    font-size: 56px;
    font-weight: 700;
    letter-spacing: 4px;
    color: var(--accent);
    transition: color 0.4s, text-shadow 0.4s;
    text-align: center;
  }
  #activity-display.critical {
    animation: flashText 0.4s infinite alternate;
    text-shadow: 0 0 40px currentColor;
  }
  @keyframes flashText {
    from { opacity: 1; }
    to { opacity: 0.3; }
  }
  .activity-time {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text2);
    margin-top: 10px;
  }
  .confidence-bar {
    width: 80%;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 14px;
    overflow: hidden;
  }
  .confidence-fill {
    height: 100%;
    width: 0%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.6s ease, background 0.4s;
  }

  /* RIGHT SIDEBAR */
  .sidebar {
    grid-column: 2;
    grid-row: 1 / 4;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* PANELS */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
  }
  .panel-title {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text2);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* STATS */
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .stat-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
  }
  .stat-value {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
  }
  .stat-label {
    font-size: 10px;
    color: var(--text2);
    letter-spacing: 1px;
    margin-top: 4px;
  }
  .stat-box.full { grid-column: 1 / -1; }

  /* CHARTS */
  .charts-row {
    grid-column: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .chart-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
  }
  .chart-wrap {
    height: 130px;
    position: relative;
  }

  /* EPISODE LOG */
  .log-panel {
    grid-column: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    max-height: 280px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .log-scroll {
    overflow-y: auto;
    flex: 1;
  }
  .log-scroll::-webkit-scrollbar { width: 4px; }
  .log-scroll::-webkit-scrollbar-track { background: transparent; }
  .log-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  thead th {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--text2);
    letter-spacing: 2px;
    padding: 6px 8px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  tbody tr {
    border-bottom: 1px solid rgba(26,42,74,0.5);
    transition: background 0.2s;
  }
  tbody tr:hover { background: var(--surface2); }
  tbody td {
    padding: 7px 8px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text);
  }
  .badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
  }
  .badge-critical { background: rgba(255,34,68,0.2); color: var(--red); }
  .badge-warning { background: rgba(255,153,0,0.2); color: var(--orange); }
  .empty-log {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text2);
    text-align: center;
    padding: 20px;
  }

  /* BUFFER BAR */
  .buffer-section {
    margin-top: 10px;
  }
  .buffer-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text2);
    margin-bottom: 5px;
  }
  .buffer-track {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
  }
  .buffer-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    transition: width 0.3s;
  }

  /* NOTIFICATION */
  #notification {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    color: var(--text);
    padding: 12px 18px;
    border-radius: 10px;
    font-family: var(--mono);
    font-size: 12px;
    opacity: 0;
    transition: opacity 0.3s;
    max-width: 320px;
    z-index: 500;
  }

  /* PATIENT CARD */
  .patient-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px;
    background: var(--surface2);
    border-radius: 10px;
    margin-bottom: 4px;
  }
  .patient-avatar {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .patient-name { font-weight: 700; font-size: 15px; }
  .patient-sub { font-family: var(--mono); font-size: 10px; color: var(--text2); }
</style>
</head>
<body>

<div id="alert-banner">&#9888; CRITICAL EVENT DETECTED &#9888;</div>

<header>
  <div class="logo">
    <div class="logo-icon">&#10084;</div>
    <div>
      <div class="logo-text">sahwa</div>
      <div class="logo-sub">NEUROLOGICAL MONITORING SYSTEM</div>
    </div>
  </div>
  <div class="header-right">
    <div class="status-pill">
      <div class="dot" id="conn-dot"></div>
      <span id="conn-text" style="font-family:var(--mono);font-size:12px;">Waiting for device</span>
    </div>
    <div class="status-pill">
      <div class="dot wrist-mode" id="mode-dot"></div>
      <span id="mode-text" style="font-family:var(--mono);font-size:12px;">WRIST MODE</span>
    </div>
    <button class="btn btn-outline" onclick="generateReport()">&#128196; REPORT</button>
  </div>
</header>

<div class="main">

  <!-- ACTIVITY DISPLAY -->
  <div class="activity-panel">
    <div class="activity-label">CURRENT ACTIVITY</div>
    <div id="activity-display">WAITING...</div>
    <div class="confidence-bar"><div class="confidence-fill" id="conf-fill"></div></div>
    <div class="activity-time" id="activity-time">--:--:--</div>
  </div>

  <!-- CHARTS ROW -->
  <div class="charts-row">
    <div class="chart-panel">
      <div class="panel-title">ACCELEROMETER (g)</div>
      <div class="chart-wrap">
        <canvas id="accelChart"></canvas>
      </div>
    </div>
    <div class="chart-panel">
      <div class="panel-title">GYROSCOPE (deg/s)</div>
      <div class="chart-wrap">
        <canvas id="gyroChart"></canvas>
      </div>
    </div>
  </div>

  <!-- EPISODE LOG -->
  <div class="log-panel">
    <div class="panel-title">EPISODE LOG</div>
    <div class="log-scroll">
      <table>
        <thead>
          <tr><th>TIME</th><th>EVENT</th><th>DURATION</th><th>SEVERITY</th></tr>
        </thead>
        <tbody id="log-body">
          <tr><td colspan="4" class="empty-log">No episodes recorded</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <!-- PATIENT CARD -->
    <div class="panel">
      <div class="panel-title">PATIENT</div>
      <div class="patient-card">
        <div class="patient-avatar">&#128100;</div>
        <div>
          <div class="patient-name" id="patient-name">Patient</div>
          <div class="patient-sub" id="patient-mode-sub">WRIST &#x2022; EPILEPSY / FALL MODE</div>
        </div>
      </div>
      <div class="buffer-section">
        <div class="buffer-label">
          <span>DATA BUFFER</span>
          <span id="buf-pct">0%</span>
        </div>
        <div class="buffer-track"><div class="buffer-fill" id="buf-fill"></div></div>
      </div>
      <div style="margin-top:10px;font-family:var(--mono);font-size:10px;color:var(--text2);">
        SAMPLES: <span id="sample-count" style="color:var(--accent);">0</span>
      </div>
    </div>

    <!-- STATS -->
    <div class="panel">
      <div class="panel-title">STATISTICS</div>
      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-value" id="stat-total">0</div>
          <div class="stat-label">EPISODES</div>
        </div>
        <div class="stat-box">
          <div class="stat-value" id="stat-longest">0s</div>
          <div class="stat-label">LONGEST</div>
        </div>
        <div class="stat-box full">
          <div class="stat-value" id="stat-frequent" style="font-size:16px;">N/A</div>
          <div class="stat-label">MOST FREQUENT</div>
        </div>
      </div>
    </div>

    <!-- CONTROLS -->
    <div class="panel">
      <div class="panel-title">CONTROLS</div>
      <div style="display:flex;flex-direction:column;gap:10px;">
        <button class="btn btn-primary" onclick="generateReport()" style="width:100%;">
          &#128196; Generate PDF Report
        </button>
        <button class="btn btn-outline" onclick="clearLog()" style="width:100%;">
          &#128465; Clear Episode Log
        </button>
      </div>
      <div style="margin-top:14px;padding:10px;background:var(--surface2);border-radius:8px;">
        <div style="font-family:var(--mono);font-size:9px;color:var(--text2);letter-spacing:2px;margin-bottom:6px;">DEVICE INSTRUCTIONS</div>
        <div style="font-size:12px;color:var(--text);line-height:1.6;">
          Short press button = switch mode<br>
          Long press (2s) = test alert
        </div>
      </div>
    </div>

  </div>
</div>

<div id="notification"></div>

<script>
const socket = io();
const POINTS = 250;

function makeChart(ctx, labels, colors) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array(POINTS).fill(''),
      datasets: labels.map((l, i) => ({
        label: l,
        data: Array(POINTS).fill(null),
        borderColor: colors[i],
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: false,
      }))
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      scales: {
        x: { display: false },
        y: {
          beginAtZero: false,
          ticks: { color: '#6080a0', font: { family: 'Share Tech Mono', size: 9 } },
          grid: { color: 'rgba(26,42,74,0.5)' },
        }
      },
      plugins: {
        legend: { labels: { color: '#6080a0', font: { family: 'Share Tech Mono', size: 9 }, boxWidth: 8 } }
      }
    }
  });
}

const accelChart = makeChart(
  document.getElementById('accelChart').getContext('2d'),
  ['X', 'Y', 'Z'],
  ['#ff4466', '#00cc66', '#00aaff']
);
const gyroChart = makeChart(
  document.getElementById('gyroChart').getContext('2d'),
  ['X', 'Y', 'Z'],
  ['#ff9900', '#cc44ff', '#00ffcc']
);

function pushChart(chart, vals) {
  chart.data.datasets.forEach((ds, i) => {
    ds.data.push(vals[i]);
    if (ds.data.length > POINTS) ds.data.shift();
  });
  chart.update('none');
}

let sampleCount = 0;
let bufferCount = 0;

socket.on('update_graphs', d => {
  pushChart(accelChart, [d.ax, d.ay, d.az]);
  pushChart(gyroChart, [d.gx, d.gy, d.gz]);
  sampleCount++;
  bufferCount = Math.min(bufferCount + 1, 256);
  document.getElementById('sample-count').textContent = sampleCount;
  const pct = Math.round((bufferCount / 256) * 100);
  document.getElementById('buf-fill').style.width = pct + '%';
  document.getElementById('buf-pct').textContent = pct + '%';
});

socket.on('update_activity', d => {
  const el = document.getElementById('activity-display');
  el.textContent = d.activity;
  el.style.color = d.color;
  el.style.textShadow = d.is_critical ? ('0 0 40px ' + d.color) : 'none';
  if (d.is_critical) {
    el.classList.add('critical');
    bufferCount = 0;
  } else {
    el.classList.remove('critical');
    bufferCount = 0;
  }
  document.getElementById('conf-fill').style.width = '90%';
  document.getElementById('conf-fill').style.background = d.color;
  document.getElementById('activity-time').textContent = new Date().toLocaleTimeString();
  setTimeout(() => {
    document.getElementById('conf-fill').style.width = '0%';
  }, 2400);
});

socket.on('update_status', d => {
  const dot = document.getElementById('conn-dot');
  const txt = document.getElementById('conn-text');
  const modeDot = document.getElementById('mode-dot');
  const modeTxt = document.getElementById('mode-text');
  const sub = document.getElementById('patient-mode-sub');
  if (d.esp_connected) {
    dot.className = 'dot connected';
    txt.textContent = 'ESP32 Connected';
  } else {
    dot.className = 'dot';
    txt.textContent = 'Waiting for device';
  }
  if (d.device_mode === 0) {
    modeDot.className = 'dot wrist-mode';
    modeTxt.textContent = 'WRIST MODE';
    sub.textContent = 'WRIST \u2022 EPILEPSY / FALL MODE';
  } else {
    modeDot.className = 'dot ankle-mode';
    modeTxt.textContent = 'ANKLE MODE';
    sub.textContent = 'ANKLE \u2022 PARKINSON\u2019S / FoG MODE';
  }
});

socket.on('update_log', d => {
  const tbody = document.getElementById('log-body');
  if (!d.log || d.log.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-log">No episodes recorded</td></tr>';
    return;
  }
  tbody.innerHTML = '';
  d.log.slice().reverse().forEach(ep => {
    const cls = ep.severity === 'CRITICAL' ? 'badge-critical' : 'badge-warning';
    const row = document.createElement('tr');
    row.innerHTML =
      '<td>' + ep.start_time + '</td>' +
      '<td>' + ep.type + '</td>' +
      '<td>' + ep.duration_seconds + 's</td>' +
      '<td><span class="badge ' + cls + '">' + ep.severity + '</span></td>';
    tbody.appendChild(row);
  });
});

socket.on('update_stats', d => {
  document.getElementById('stat-total').textContent = d.total;
  document.getElementById('stat-longest').textContent = d.longest + 's';
  document.getElementById('stat-frequent').textContent = d.most_frequent;
});

socket.on('critical_alert', d => {
  const banner = document.getElementById('alert-banner');
  banner.textContent = '\u26a0 CRITICAL: ' + d.type + ' DETECTED AT ' + d.time + ' \u26a0';
  banner.style.display = 'block';
  setTimeout(() => { banner.style.display = 'none'; }, 12000);
});

socket.on('show_notification', d => {
  showNotif(d.message);
});

socket.on('connect', () => {
  showNotif('Connected to sahwa server');
});

function showNotif(msg) {
  const el = document.getElementById('notification');
  el.textContent = msg;
  el.style.opacity = 1;
  setTimeout(() => { el.style.opacity = 0; }, 3500);
}

function generateReport() {
  window.location.href = '/report';
}

function clearLog() {
  if (confirm('Clear all episode records?')) {
    fetch('/clear_log', { method: 'POST' })
      .then(() => showNotif('Episode log cleared'));
  }
}

// Set patient name
fetch('/patient_name').then(r => r.json()).then(d => {
  document.getElementById('patient-name').textContent = d.name;
});
</script>
</body>
</html>"""

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/patient_name")
def patient_name():
    return jsonify({"name": PATIENT_NAME})


@app.route("/report")
def report():
    pdf = generate_pdf()
    if pdf is None:
        return "PDF generation failed. Install reportlab.", 500
    filename = "sahwa_Report_" + datetime.now().strftime("%Y%m%d_%H%M") + ".pdf"
    resp = make_response(pdf.getvalue())
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = "attachment; filename=" + filename
    return resp


@app.route("/clear_log", methods=["POST"])
def clear_log():
    global episode_log, current_episode, normal_streak
    episode_log = []
    current_episode = None
    normal_streak = 0
    if os.path.exists(EPISODES_FILE):
        os.remove(EPISODES_FILE)
    socketio.emit("update_log", {"log": []})
    socketio.emit("update_stats", {"total": 0, "longest": 0, "most_frequent": "N/A"})
    return jsonify({"ok": True})


@socketio.on("connect")
def on_connect():
    with data_lock:
        s = dict(state)
    socketio.emit("update_status", {
        "esp_connected": s["esp_connected"],
        "device_mode": s["device_mode"],
        "ip": "",
    }, room=None)
    socketio.emit("update_activity", {
        "activity": s["prediction"],
        "color": s["color"],
        "is_critical": s["is_critical"],
        "label": s["pred_label"],
    }, room=None)
    socketio.emit("update_log", {"log": get_log_json()}, room=None)
    socketio.emit("update_stats", get_stats(), room=None)

# ============================================================
# MAIN
# ============================================================
def load_episodes():
    global episode_log
    if os.path.exists(EPISODES_FILE):
        try:
            with open(EPISODES_FILE, "r") as f:
                episode_log = json.load(f)
            print("[Main] Loaded " + str(len(episode_log)) + " previous episodes")
        except Exception:
            episode_log = []


def main():
    print("=" * 60)
    print("  sahwa Monitoring System")
    print("  Version 1.0 | Abu Dhabi University URIC 2026")
    print("=" * 60)

    load_episodes()
    models = load_models()

    tcp = TCPServer(models)
    tcp.start()

    engine = InferenceEngine(models)
    engine.start()

    url = "http://127.0.0.1:" + str(WEB_PORT)
    print("[Web] Dashboard: " + url)
    print("[TCP] Waiting for ESP32 on port " + str(TCP_PORT))
    print("[Info] Flash sahwa_stream.ino and enable hotspot 'ysfthecreator'")
    print("=" * 60)

    threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    socketio.run(app, host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
