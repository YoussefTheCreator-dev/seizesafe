# -*- coding: utf-8 -*-
"""
Sahwa Server v2.0
Real-Time Neurological Event Monitoring System
"""

import os
import sys
import json
import threading
import time
import smtplib
import io
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import joblib
from flask import Flask, render_template_string, make_response, jsonify, request
from flask_socketio import SocketIO

# ============================================================
# CONFIGURATION
# ============================================================
WEB_PORT = 5000

# ── CREDENTIALS via environment variables ──────────────────
# On Railway: set these in the Variables tab
# Locally: create a .env file (see .env.example)
CAREGIVER_EMAIL    = os.environ.get("CAREGIVER_EMAIL",    "caregiver@example.com")
GMAIL_SENDER       = os.environ.get("GMAIL_SENDER",       "your.gmail@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "your-app-password")

WINDOW_SIZE = 256
SAMPLE_RATE = 50
EPISODES_FILE = "episodes.json"
SMOOTHING_WINDOW = 3

WRIST_LABELS  = {0:"Stand",1:"Walk",2:"FastWalk",3:"Sit",4:"SitStand",6:"Fall",8:"Seizure"}
ANKLE_LABELS  = {0:"Stand",1:"Walk",2:"FastWalk",3:"Sit",4:"SitStand",5:"Stairs",7:"FoG"}
CRITICAL_WRIST = {6, 8}
CRITICAL_ANKLE = {7}
WARNING_LABELS = {"FOG","SITSTAND"}

# ============================================================
# SHARED STATE
# ============================================================
data_lock    = threading.Lock()
imu_buffer   = deque()
# esp_sock_ref removed (WebSocket replaces TCP)

state = {
    "device_mode":      0,
    "esp_connected":    False,
    "prediction":       "Waiting for device...",
    "pred_label":       -1,
    "color":            "#00aaff",
    "is_critical":      False,
    "samples":          0,
    "last_update":      "",
    "patient_name":     "Patient",
    "esp_ip":           "",
    "caregiver_email":  "",
}

episode_log    = []
current_episode = None
normal_streak  = 0

# ============================================================
# FLASK
# ============================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = "sahwa2026"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ============================================================
# FEATURE EXTRACTION
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
    df["gyro_magnitude"]  = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    signals = ["ax","ay","az","gx","gy","gz","accel_magnitude","gyro_magnitude"]
    feats = []
    for sig in signals:
        d = df[sig].values.astype(float)
        feats += [float(np.mean(d)), float(np.std(d)),
                  float(np.min(d)),  float(np.max(d)),
                  float(np.sqrt(np.mean(d**2)))]
        centered = d - np.mean(d)
        zcr = np.where(np.diff(np.sign(centered)))[0]
        feats.append(float(len(zcr) / len(d)))
        n   = len(d)
        yf  = fft(d)
        mags  = np.abs(yf[1:n//2])
        freqs = fftfreq(n, 1.0/SAMPLE_RATE)[1:n//2]
        if len(mags) > 0:
            feats.append(float(freqs[np.argmax(mags)]))
            feats.append(spectral_entropy(mags**2))
        else:
            feats += [0.0, 0.0]
    return np.array(feats).reshape(1, -1)

# ============================================================
# MODEL
# ============================================================
class ModelPack:
    def __init__(self, model_path, scaler_path, mapping_path, critical_set, label_names):
        self.model  = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(mapping_path) as f:
            raw = json.load(f)
        self.mapping     = {int(k): v for k,v in raw.items()}
        self.critical    = critical_set
        self.label_names = label_names

    def predict(self, df):
        feats      = extract_features(df)
        scaled     = self.scaler.transform(feats)
        pred_idx   = int(self.model.predict(scaled)[0])
        orig_label = next((k for k,v in self.mapping.items() if v == pred_idx), -1)
        name       = self.label_names.get(orig_label, "Unknown")
        is_crit    = orig_label in self.critical
        return orig_label, name, is_crit


def load_models():
    models = {}
    wf = ["wrist_xgb_model.pkl","wrist_xgb_scaler.pkl","wrist_label_mapping.json"]
    af = ["ankle_rf_model.pkl","ankle_rf_scaler.pkl","ankle_label_mapping.json"]
    if all(os.path.exists(f) for f in wf):
        models[0] = ModelPack("wrist_xgb_model.pkl","wrist_xgb_scaler.pkl",
                               "wrist_label_mapping.json", CRITICAL_WRIST, WRIST_LABELS)
        print("[Models] Wrist XGBoost loaded")
    else:
        print("[Models] WARNING: Wrist model missing")
    if all(os.path.exists(f) for f in af):
        models[1] = ModelPack("ankle_rf_model.pkl","ankle_rf_scaler.pkl",
                               "ankle_label_mapping.json", CRITICAL_ANKLE, ANKLE_LABELS)
        print("[Models] Ankle RF loaded")
    else:
        print("[Models] WARNING: Ankle model missing")
    if not models:
        print("[Models] ERROR: No models found"); sys.exit(1)
    return models

# ============================================================
# EPISODE MANAGEMENT
# ============================================================
def start_episode(event_type):
    global current_episode
    current_episode = {
        "type": event_type,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": 0,
        "severity": "CRITICAL" if event_type in ["SEIZURE","FALL"] else "WARNING",
    }

def end_episode():
    global current_episode, episode_log
    if not current_episode:
        return
    current_episode["end_time"] = datetime.now().isoformat()
    s = datetime.fromisoformat(current_episode["start_time"])
    e = datetime.fromisoformat(current_episode["end_time"])
    current_episode["duration_seconds"] = int((e - s).total_seconds())
    episode_log.append(current_episode)
    save_episodes()
    socketio.emit("update_log",   {"log": get_log_json()})
    socketio.emit("update_stats", get_stats())
    current_episode = None

def save_episodes():
    with open(EPISODES_FILE,"w") as f:
        json.dump(episode_log, f, indent=2)

def get_log_json():
    result = []
    for ep in episode_log[-30:]:
        result.append({
            "start_time":       ep["start_time"][:19].replace("T"," "),
            "type":             ep["type"],
            "duration_seconds": ep["duration_seconds"],
            "severity":         ep["severity"],
        })
    return result

def get_stats():
    if not episode_log:
        return {"total":0,"longest":0,"most_frequent":"N/A"}
    types     = [ep["type"] for ep in episode_log]
    most_freq = max(set(types), key=types.count)
    longest   = max(ep["duration_seconds"] for ep in episode_log)
    return {"total": len(episode_log), "longest": longest, "most_frequent": most_freq}

# ============================================================
# ALERTS
# ============================================================
def trigger_alert(event_type):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[ALERT] " + event_type + " at " + now)
    socketio.emit("critical_alert", {"type": event_type, "time": now})
    send_buzzer()
    threading.Thread(target=send_email, args=(event_type, now), daemon=True).start()

def send_buzzer():
    # Send ALERT to ESP32 via WebSocket instead of TCP
    socketio.emit("device_command", {"cmd": "ALERT"}, namespace="/esp32")

def send_email(event_type, timestamp):
    to_email = state.get("caregiver_email","").strip()
    if not to_email:
        print("[Email] Skipped — no caregiver email configured")
        return
    if GMAIL_APP_PASSWORD == "your-app-password":
        print("[Email] Skipped — no Gmail app password configured")
        return
    try:
        pname = state.get("patient_name","Patient")
        msg = MIMEMultipart()
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = to_email
        msg["Subject"] = "SAHWA ALERT: " + event_type + " detected"
        body = ("Sahwa Emergency Alert\n\n"
                "Patient: " + pname + "\n"
                "Event: "   + event_type + "\n"
                "Time: "    + timestamp  + "\n\n"
                "Please check on your patient immediately.\n\n"
                "-- Sahwa Monitoring System")
        msg.attach(MIMEText(body,"plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            s.sendmail(GMAIL_SENDER, to_email, msg.as_string())
        print("[Email] Alert sent to " + to_email)
    except Exception as e:
        print("[Email] Failed: " + str(e))

# ============================================================
# INFERENCE ENGINE
# ============================================================
class InferenceEngine(threading.Thread):
    def __init__(self, models):
        super().__init__(daemon=True)
        self.models      = models
        self.running     = True
        self.pred_history = []

    def run(self):
        global normal_streak, current_episode
        print("[Inference] Engine started")
        while self.running:
            with data_lock:
                buf_len = len(imu_buffer)
            if buf_len >= WINDOW_SIZE:
                with data_lock:
                    window = [imu_buffer.popleft() for _ in range(WINDOW_SIZE)]
                try:
                    df   = pd.DataFrame(window)
                    mode = state["device_mode"]
                    pack = self.models.get(mode)
                    if pack is None:
                        time.sleep(0.1)
                        continue

                    orig_label, name, is_crit_raw = pack.predict(df)

                    # Temporal smoothing — require N consecutive same predictions
                    self.pred_history.append(orig_label)
                    if len(self.pred_history) > SMOOTHING_WINDOW:
                        self.pred_history.pop(0)
                    confirmed = (
                        len(self.pred_history) == SMOOTHING_WINDOW
                        and all(p == orig_label for p in self.pred_history)
                    )
                    is_crit = is_crit_raw and confirmed

                    name_upper = name.upper()
                    if is_crit:
                        color = "#ff2244"
                    elif name_upper in WARNING_LABELS:
                        color = "#ff9900"
                    else:
                        color = "#00cc66"

                    with data_lock:
                        state.update({
                            "prediction":  name_upper,
                            "pred_label":  orig_label,
                            "color":       color,
                            "is_critical": is_crit,
                            "last_update": datetime.now().strftime("%H:%M:%S"),
                        })

                    socketio.emit("update_activity", {
                        "activity":    name_upper,
                        "color":       color,
                        "is_critical": is_crit,
                        "label":       orig_label,
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

# TCPServer removed — ESP32 now connects via WebSocket /esp32 namespace

# ============================================================
# INFERENCE ENGINE
# ============================================================
class InferenceEngine(threading.Thread):
    def __init__(self, models):
        super().__init__(daemon=True)
        self.models      = models
        self.running     = True
        self.pred_history = []

    def run(self):
        global normal_streak, current_episode
        print("[Inference] Engine started")
        while self.running:
            with data_lock:
                buf_len = len(imu_buffer)
            if buf_len >= WINDOW_SIZE:
                with data_lock:
                    window = [imu_buffer.popleft() for _ in range(WINDOW_SIZE)]
                try:
                    df   = pd.DataFrame(window)
                    mode = state["device_mode"]
                    pack = self.models.get(mode)
                    if pack is None:
                        time.sleep(0.1)
                        continue

                    orig_label, name, is_crit_raw = pack.predict(df)

                    # Temporal smoothing — require N consecutive same predictions
                    self.pred_history.append(orig_label)
                    if len(self.pred_history) > SMOOTHING_WINDOW:
                        self.pred_history.pop(0)
                    confirmed = (
                        len(self.pred_history) == SMOOTHING_WINDOW
                        and all(p == orig_label for p in self.pred_history)
                    )
                    is_crit = is_crit_raw and confirmed

                    name_upper = name.upper()
                    if is_crit:
                        color = "#ff2244"
                    elif name_upper in WARNING_LABELS:
                        color = "#ff9900"
                    else:
                        color = "#00cc66"

                    with data_lock:
                        state.update({
                            "prediction":  name_upper,
                            "pred_label":  orig_label,
                            "color":       color,
                            "is_critical": is_crit,
                            "last_update": datetime.now().strftime("%H:%M:%S"),
                        })

                    socketio.emit("update_activity", {
                        "activity":    name_upper,
                        "color":       color,
                        "is_critical": is_crit,
                        "label":       orig_label,
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
# TCP SERVER
# ============================================================
# TCPServer removed — ESP32 now uses WebSocket namespace /esp32
# ============================================================
# PDF REPORT
# ============================================================
def generate_pdf(patient_name):
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
        elements.append(Paragraph("Sahwa Monitoring Report", styles["Title"]))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["Normal"]))
        elements.append(Paragraph("Patient: " + patient_name, styles["Normal"]))
        elements.append(Spacer(1, 16))

        stats = get_stats()
        elements.append(Paragraph("Summary", styles["Heading2"]))
        t = Table([
            ["Metric","Value"],
            ["Total Episodes", str(stats["total"])],
            ["Longest Episode", str(stats["longest"]) + "s"],
            ["Most Frequent", stats["most_frequent"]],
        ], colWidths=[3*inch, 3*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#0a1020")),
            ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5f5")]),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 16))

        elements.append(Paragraph("Episode Log", styles["Heading2"]))
        if episode_log:
            rows = [["Time","Event","Duration","Severity"]]
            for ep in reversed(episode_log):
                rows.append([
                    ep["start_time"][:19].replace("T"," "),
                    ep["type"],
                    str(ep["duration_seconds"]) + "s",
                    ep["severity"],
                ])
            t2 = Table(rows, colWidths=[2*inch,1.5*inch,1*inch,1.5*inch])
            t2.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0a1020")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("ALIGN",(0,0),(-1,-1),"CENTER"),
                ("GRID",(0,0),(-1,-1),0.5,colors.grey),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f5f5")]),
            ]))
            elements.append(t2)
        else:
            elements.append(Paragraph("No episodes recorded.", styles["Normal"]))

        elements.append(Spacer(1,16))
        elements.append(Paragraph("Recommendations", styles["Heading2"]))
        n = stats["total"]
        if n == 0:
            rec = "No critical events detected. Continue routine monitoring."
        elif n < 3:
            rec = str(n) + " episode(s) detected. Monitor closely and consult your neurologist."
        else:
            rec = (str(n) + " episodes detected. Urgent clinical review recommended. "
                   "Consider medication timing adjustment.")
        elements.append(Paragraph(rec, styles["Normal"]))
        elements.append(Spacer(1, 24))
        elements.append(Paragraph("-- Sahwa Automated Report --", styles["Normal"]))
        doc.build(elements)
        buf.seek(0)
        return buf
    except Exception as e:
        print("[PDF] Error: " + str(e))
        return None

# ============================================================
# LOGIN PAGE HTML
# ============================================================
LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Sahwa — Setup</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #040810;
  --surface: #0a1020;
  --surface2: #0f1830;
  --border: #1a2a4a;
  --accent: #00aaff;
  --accent2: #00ffcc;
  --text: #c8d8f0;
  --text2: #6080a0;
  --mono: 'Share Tech Mono', monospace;
  --sans: 'Rajdhani', sans-serif;
}
html, body {
  height: 100%;
  background: var(--bg);
  font-family: var(--sans);
  color: var(--text);
  overflow: hidden;
}

/* ANIMATED BACKGROUND */
.bg-canvas {
  position: fixed; inset: 0; z-index: 0;
  overflow: hidden;
}
.bg-line {
  position: absolute;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,170,255,0.15), transparent);
  animation: scanline 4s linear infinite;
  opacity: 0;
}
@keyframes scanline {
  0%   { transform: translateY(-20px); opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  100% { transform: translateY(100vh); opacity: 0; }
}
.grid-bg {
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(0,170,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,170,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  z-index: 0;
}
.glow-orb {
  position: fixed;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.12;
  z-index: 0;
  animation: orb 8s ease-in-out infinite alternate;
}
.orb1 { width:500px;height:500px;background:#00aaff;left:-100px;top:-100px; }
.orb2 { width:400px;height:400px;background:#00ffcc;right:-80px;bottom:-80px; animation-delay:-4s; }
@keyframes orb { from { transform: scale(1) rotate(0deg); } to { transform: scale(1.2) rotate(15deg); } }

/* CARD */
.center {
  position: relative; z-index: 10;
  min-height: 100vh;
  display: flex; align-items: center; justify-content: center;
}
.card {
  background: rgba(10,16,32,0.92);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 48px 44px;
  width: 460px;
  backdrop-filter: blur(20px);
  box-shadow: 0 0 60px rgba(0,170,255,0.08), 0 40px 80px rgba(0,0,0,0.5);
  animation: cardIn 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes cardIn {
  from { opacity:0; transform:translateY(30px) scale(0.96); }
  to   { opacity:1; transform:translateY(0) scale(1); }
}

/* LOGO */
.logo {
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 32px;
}
.logo-icon {
  width: 52px; height: 52px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 26px;
  box-shadow: 0 0 30px rgba(0,170,255,0.3);
  animation: heartbeat 2s ease-in-out infinite;
}
@keyframes heartbeat {
  0%,100% { transform: scale(1); }
  14% { transform: scale(1.1); }
  28% { transform: scale(1); }
  42% { transform: scale(1.06); }
  70% { transform: scale(1); }
}
.logo-main {
  font-size: 28px; font-weight: 700; letter-spacing: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.logo-sub {
  font-family: var(--mono); font-size: 9px; color: var(--text2);
  letter-spacing: 2px; margin-top: 2px;
}

/* WELCOME TEXT */
.welcome {
  font-family: var(--mono); font-size: 11px;
  color: var(--text2); letter-spacing: 3px;
  margin-bottom: 28px; padding-bottom: 20px;
  border-bottom: 1px solid var(--border);
}

/* FORM */
.field { margin-bottom: 20px; }
.field-label {
  font-family: var(--mono); font-size: 10px;
  color: var(--text2); letter-spacing: 2px;
  margin-bottom: 8px; display: flex; align-items: center; gap: 6px;
}
.field-label .required { color: var(--accent); }
.field input {
  width: 100%;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 13px 16px;
  font-family: var(--mono); font-size: 14px;
  color: var(--text);
  outline: none;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.field input::placeholder { color: var(--text2); }
.field input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(0,170,255,0.12);
}

.ip-note {
  background: rgba(0,170,255,0.06);
  border: 1px solid rgba(0,170,255,0.15);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 12px;
  color: var(--text2);
  margin-top: 8px;
  line-height: 1.5;
}
.ip-note strong { color: var(--accent); }

/* SUBMIT */
.btn-submit {
  width: 100%;
  padding: 16px;
  background: linear-gradient(135deg, var(--accent), #0066cc);
  border: none; border-radius: 12px;
  color: white;
  font-family: var(--sans); font-weight: 700;
  font-size: 15px; letter-spacing: 2px;
  cursor: pointer;
  margin-top: 8px;
  transition: all 0.3s;
  position: relative; overflow: hidden;
}
.btn-submit::after {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
  opacity: 0; transition: opacity 0.3s;
}
.btn-submit:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,170,255,0.4); }
.btn-submit:hover::after { opacity: 1; }
.btn-submit:active { transform: translateY(0); }

/* MODE SELECTOR */
.mode-section { margin-bottom: 20px; }
.mode-label {
  font-family: var(--mono); font-size: 10px;
  color: var(--text2); letter-spacing: 2px;
  margin-bottom: 10px;
}
.mode-options {
  display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
}
.mode-opt {
  background: var(--surface2);
  border: 2px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  cursor: pointer;
  transition: all 0.25s;
  text-align: center;
}
.mode-opt:hover { border-color: rgba(0,170,255,0.4); }
.mode-opt.selected { border-color: var(--accent); background: rgba(0,170,255,0.08); }
.mode-opt.ankle-selected { border-color: #cc44ff; background: rgba(204,68,255,0.08); }
.mode-opt-icon { font-size: 22px; margin-bottom: 6px; }
.mode-opt-name { font-weight: 700; font-size: 13px; letter-spacing: 1px; }
.mode-opt-desc { font-family: var(--mono); font-size: 9px; color: var(--text2); margin-top: 3px; }

/* VERSION */
.version {
  text-align: center; margin-top: 20px;
  font-family: var(--mono); font-size: 9px; color: var(--text2);
  letter-spacing: 2px;
}
</style>
</head>
<body>
<div class="grid-bg"></div>
<div class="glow-orb orb1"></div>
<div class="glow-orb orb2"></div>
<div class="bg-canvas" id="bgCanvas"></div>

<div class="center">
  <div class="card">
    <div class="logo">
      <div class="logo-icon">&#10084;</div>
      <div>
        <div class="logo-main">SAHWA</div>
        <div class="logo-sub">NEUROLOGICAL MONITORING SYSTEM</div>
      </div>
    </div>

    <div class="welcome">PATIENT SETUP &mdash; ENTER DETAILS TO BEGIN MONITORING</div>

    <div class="field">
      <div class="field-label">PATIENT NAME <span class="required">*</span></div>
      <input type="text" id="patient-name" placeholder="Enter patient name" autocomplete="off">
    </div>

    <div class="field">
      <div class="field-label">CAREGIVER EMAIL <span style="color:var(--text2);font-size:9px;">(optional)</span></div>
      <input type="email" id="caregiver-email" placeholder="caregiver@example.com" autocomplete="off">
      <div class="ip-note">
        Emergency alerts will be sent here when a critical event is detected.<br>
        Leave blank to disable email alerts.
      </div>
    </div>

    <div class="field">
      <div class="field-label">ESP32 IP ADDRESS <span class="required">*</span></div>
      <input type="text" id="esp-ip" placeholder="e.g. 192.168.137.xxx" autocomplete="off">
      <div class="ip-note">
        Power on the device and check the OLED display.<br>
        The IP shown under <strong>WiFi Connected</strong> is your device IP.<br>
        The server listens on <strong>192.168.137.1:8888</strong> — your laptop hotspot must be active.
      </div>
    </div>

    <div class="mode-section">
      <div class="mode-label">INITIAL DETECTION MODE</div>
      <div class="mode-options">
        <div class="mode-opt selected" id="mode-wrist" onclick="selectMode(0)">
          <div class="mode-opt-icon">&#9785;</div>
          <div class="mode-opt-name" style="color:#00ccff">WRIST</div>
          <div class="mode-opt-desc">SEIZURE + FALL</div>
        </div>
        <div class="mode-opt" id="mode-ankle" onclick="selectMode(1)">
          <div class="mode-opt-icon">&#128694;</div>
          <div class="mode-opt-name" style="color:#cc44ff">ANKLE</div>
          <div class="mode-opt-desc">FOG + GAIT</div>
        </div>
      </div>
    </div>

    <button class="btn-submit" onclick="startMonitoring()">
      &#9654;&nbsp;&nbsp;START MONITORING
    </button>

    <div class="version">Sahwa v2.0 &bull; ABU DHABI UNIVERSITY URIC 2026</div>
  </div>
</div>

<script>
let selectedMode = 0;

function selectMode(m) {
  selectedMode = m;
  document.getElementById('mode-wrist').className = 'mode-opt' + (m===0 ? ' selected' : '');
  document.getElementById('mode-ankle').className = 'mode-opt' + (m===1 ? ' ankle-selected' : '');
}

function startMonitoring() {
  const name  = document.getElementById('patient-name').value.trim();
  const email = document.getElementById('caregiver-email').value.trim();
  const ip    = document.getElementById('esp-ip').value.trim();
  if (!name) { document.getElementById('patient-name').focus(); return; }
  if (!ip)   { document.getElementById('esp-ip').focus(); return; }
  window.location.href = '/dashboard?name=' + encodeURIComponent(name)
                        + '&email=' + encodeURIComponent(email)
                        + '&ip='   + encodeURIComponent(ip)
                        + '&mode=' + selectedMode;
}

// Animated scan lines
const canvas = document.getElementById('bgCanvas');
for (let i = 0; i < 6; i++) {
  const line = document.createElement('div');
  line.className = 'bg-line';
  line.style.width = (60 + Math.random() * 40) + '%';
  line.style.left  = (Math.random() * 40) + '%';
  line.style.animationDuration = (3 + Math.random() * 4) + 's';
  line.style.animationDelay    = (Math.random() * 4) + 's';
  canvas.appendChild(line);
}

document.addEventListener('keydown', e => {
  if (e.key === 'Enter') startMonitoring();
});
</script>
</body>
</html>"""

# ============================================================
# DASHBOARD HTML
# ============================================================
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Sahwa Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#040810;--surface:#0a1020;--surface2:#0f1830;
  --border:#1a2a4a;--accent:#00aaff;--accent2:#00ffcc;
  --green:#00cc66;--orange:#ff9900;--red:#ff2244;
  --text:#c8d8f0;--text2:#6080a0;
  --mono:'Share Tech Mono',monospace;--sans:'Rajdhani',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden;}
body::before{
  content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse at 20% 20%,rgba(0,170,255,0.04) 0%,transparent 60%),
             radial-gradient(ellipse at 80% 80%,rgba(0,255,204,0.03) 0%,transparent 60%);
  pointer-events:none;z-index:0;
}
.grid-bg{
  position:fixed;inset:0;
  background-image:linear-gradient(rgba(0,170,255,0.02) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,170,255,0.02) 1px,transparent 1px);
  background-size:40px 40px;z-index:0;
}

/* ALERT BANNER */
#alert-banner{
  display:none;position:fixed;top:0;left:0;right:0;
  background:var(--red);color:white;text-align:center;
  padding:14px;font-size:17px;font-weight:700;letter-spacing:3px;
  z-index:1000;animation:flashBanner 0.5s infinite alternate;
}
@keyframes flashBanner{from{opacity:1;}to{opacity:0.5;}}

/* HEADER */
header{
  position:relative;z-index:10;
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 28px;
  background:rgba(10,16,32,0.95);
  border-bottom:1px solid var(--border);
  backdrop-filter:blur(10px);
}
.logo{display:flex;align-items:center;gap:12px;}
.logo-icon{
  width:36px;height:36px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-size:18px;animation:heartbeat 2s ease-in-out infinite;
}
@keyframes heartbeat{
  0%,100%{transform:scale(1);}14%{transform:scale(1.12);}
  28%{transform:scale(1);}42%{transform:scale(1.06);}70%{transform:scale(1);}
}
.logo-text{font-size:22px;font-weight:700;letter-spacing:2px;
  background:linear-gradient(90deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.logo-sub{font-family:var(--mono);font-size:9px;color:var(--text2);letter-spacing:1px;}

.header-center{display:flex;align-items:center;gap:12px;}
.status-pill{
  display:flex;align-items:center;gap:8px;padding:5px 12px;
  border-radius:20px;border:1px solid var(--border);
  font-family:var(--mono);font-size:11px;background:var(--surface2);
}
.dot{width:7px;height:7px;border-radius:50%;background:var(--text2);}
.dot.on{background:var(--green);box-shadow:0 0 8px var(--green);animation:blink 1.5s infinite;}
.dot.wrist{background:#00ccff;box-shadow:0 0 8px #00ccff;}
.dot.ankle{background:#cc44ff;box-shadow:0 0 8px #cc44ff;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:0.3;}}

/* MODE TOGGLE */
.mode-toggle{
  display:flex;align-items:center;gap:0;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;overflow:hidden;
}
.mode-btn{
  padding:7px 16px;font-family:var(--mono);font-size:11px;
  letter-spacing:1px;cursor:pointer;border:none;
  background:transparent;color:var(--text2);transition:all 0.25s;
}
.mode-btn.active-wrist{background:rgba(0,204,255,0.15);color:#00ccff;}
.mode-btn.active-ankle{background:rgba(204,68,255,0.15);color:#cc44ff;}
.mode-btn:hover{color:var(--text);}

.header-right{display:flex;align-items:center;gap:10px;}
.btn{
  padding:7px 16px;border:none;border-radius:8px;
  font-family:var(--sans);font-weight:600;font-size:12px;
  cursor:pointer;letter-spacing:1px;transition:all 0.2s;
}
.btn-primary{background:linear-gradient(135deg,var(--accent),#0066cc);color:white;}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,170,255,0.4);}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text);}
.btn-outline:hover{border-color:var(--accent);color:var(--accent);}
.btn-back{background:transparent;border:1px solid var(--border);color:var(--text2);font-size:11px;}
.btn-back:hover{color:var(--text);}

/* MAIN */
.main{
  position:relative;z-index:1;
  display:grid;
  grid-template-columns:1fr 300px;
  grid-template-rows:auto auto 1fr;
  gap:14px;padding:18px 24px;
  min-height:calc(100vh - 66px);
}

/* ACTIVITY */
.activity-panel{
  grid-column:1;
  background:var(--surface);border:1px solid var(--border);
  border-radius:16px;display:flex;flex-direction:column;
  align-items:center;justify-content:center;padding:28px;
  min-height:190px;position:relative;overflow:hidden;
}
.activity-panel::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at center,rgba(0,170,255,0.05) 0%,transparent 70%);
  pointer-events:none;transition:background 0.5s;
}
.activity-panel.critical-bg::before{
  background:radial-gradient(ellipse at center,rgba(255,34,68,0.08) 0%,transparent 70%);
}
.activity-label{font-family:var(--mono);font-size:10px;color:var(--text2);letter-spacing:3px;margin-bottom:10px;}
#activity-display{
  font-family:var(--sans);font-size:52px;font-weight:700;
  letter-spacing:4px;color:var(--accent);
  transition:color 0.4s,text-shadow 0.4s;text-align:center;
}
#activity-display.flashing{animation:flashText 0.4s infinite alternate;text-shadow:0 0 40px currentColor;}
@keyframes flashText{from{opacity:1;}to{opacity:0.25;}}
.conf-bar{width:70%;height:3px;background:var(--border);border-radius:2px;margin-top:12px;overflow:hidden;}
.conf-fill{height:100%;width:0%;background:var(--accent);border-radius:2px;transition:width 0.6s,background 0.4s;}
.activity-time{font-family:var(--mono);font-size:10px;color:var(--text2);margin-top:8px;}

/* CHARTS */
.charts-row{grid-column:1;display:grid;grid-template-columns:1fr 1fr;gap:14px;}
.chart-panel{
  background:var(--surface);border:1px solid var(--border);
  border-radius:14px;padding:16px;
}
.panel-title{
  font-family:var(--mono);font-size:9px;color:var(--text2);
  letter-spacing:3px;margin-bottom:12px;display:flex;align-items:center;gap:8px;
}
.panel-title::after{content:'';flex:1;height:1px;background:var(--border);}
.chart-wrap{height:120px;position:relative;}

/* LOG */
.log-panel{
  grid-column:1;background:var(--surface);border:1px solid var(--border);
  border-radius:14px;padding:16px;max-height:260px;
  overflow:hidden;display:flex;flex-direction:column;
}
.log-scroll{overflow-y:auto;flex:1;}
.log-scroll::-webkit-scrollbar{width:3px;}
.log-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
table{width:100%;border-collapse:collapse;font-size:11px;}
thead th{
  font-family:var(--mono);font-size:9px;color:var(--text2);
  letter-spacing:2px;padding:5px 8px;text-align:left;
  border-bottom:1px solid var(--border);
}
tbody tr{border-bottom:1px solid rgba(26,42,74,0.4);transition:background 0.15s;}
tbody tr:hover{background:var(--surface2);}
tbody td{padding:6px 8px;font-family:var(--mono);font-size:10px;}
.badge{padding:2px 8px;border-radius:4px;font-size:9px;font-weight:700;}
.badge-critical{background:rgba(255,34,68,0.2);color:var(--red);}
.badge-warning{background:rgba(255,153,0,0.2);color:var(--orange);}

/* SIDEBAR */
.sidebar{grid-column:2;grid-row:1/4;display:flex;flex-direction:column;gap:14px;}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px;}

/* PATIENT CARD */
.patient-card{
  display:flex;align-items:center;gap:12px;
  padding:12px;background:var(--surface2);border-radius:10px;margin-bottom:8px;
}
.patient-avatar{
  width:38px;height:38px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;font-size:18px;
}
.patient-name-text{font-weight:700;font-size:15px;}
.patient-sub{font-family:var(--mono);font-size:9px;color:var(--text2);}
.buf-label{display:flex;justify-content:space-between;font-family:var(--mono);font-size:9px;color:var(--text2);margin-bottom:4px;margin-top:10px;}
.buf-track{height:3px;background:var(--border);border-radius:2px;overflow:hidden;}
.buf-fill{height:100%;width:0%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:2px;transition:width 0.3s;}
.sample-count{font-family:var(--mono);font-size:9px;color:var(--text2);margin-top:6px;}

/* STATS */
.stats-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.stat-box{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:10px;text-align:center;}
.stat-val{font-family:var(--mono);font-size:20px;font-weight:700;color:var(--accent);}
.stat-lab{font-size:9px;color:var(--text2);letter-spacing:1px;margin-top:3px;}
.stat-full{grid-column:1/-1;}

/* NOTIFICATION */
#notification{
  position:fixed;bottom:20px;right:20px;
  background:var(--surface2);border:1px solid var(--border);
  color:var(--text);padding:11px 16px;border-radius:10px;
  font-family:var(--mono);font-size:11px;
  opacity:0;transition:opacity 0.3s;max-width:300px;z-index:500;
  border-left-width:3px;
}
.notif-success{border-left-color:var(--green)!important;}
.notif-warning{border-left-color:var(--orange)!important;}
.notif-info{border-left-color:var(--accent)!important;}
.notif-critical{border-left-color:var(--red)!important;}

/* DEVICE IP DISPLAY */
.ip-badge{
  font-family:var(--mono);font-size:9px;color:var(--text2);
  background:var(--surface2);border:1px solid var(--border);
  padding:3px 8px;border-radius:5px;
}
</style>
</head>
<body>
<div class="grid-bg"></div>

<div id="alert-banner">&#9888; CRITICAL: <span id="alert-event"></span> DETECTED &#9888;</div>

<header>
  <div class="logo">
    <div class="logo-icon">&#10084;</div>
    <div>
      <div class="logo-text">SAHWA</div>
      <div class="logo-sub">NEUROLOGICAL MONITORING SYSTEM</div>
    </div>
  </div>

  <div class="header-center">
    <div class="status-pill">
      <div class="dot" id="conn-dot"></div>
      <span id="conn-text">Waiting for device</span>
    </div>
    <div class="mode-toggle">
      <button class="mode-btn active-wrist" id="btn-wrist" onclick="switchMode(0)">&#9785; WRIST</button>
      <button class="mode-btn" id="btn-ankle" onclick="switchMode(1)">&#128694; ANKLE</button>
    </div>
    <div class="ip-badge" id="esp-ip-badge" style="display:none">ESP: <span id="esp-ip-val"></span></div>
  </div>

  <div class="header-right">
    <button class="btn btn-outline" onclick="generateReport()">&#128196; REPORT</button>
    <button class="btn btn-back" onclick="window.location.href='/'">&#8592; SETUP</button>
  </div>
</header>

<div class="main">

  <div class="activity-panel" id="activity-panel">
    <div class="activity-label">CURRENT ACTIVITY</div>
    <div id="activity-display">WAITING...</div>
    <div class="conf-bar"><div class="conf-fill" id="conf-fill"></div></div>
    <div class="activity-time" id="activity-time">--:--:--</div>
  </div>

  <div class="charts-row">
    <div class="chart-panel">
      <div class="panel-title">ACCELEROMETER (g)</div>
      <div class="chart-wrap"><canvas id="accelChart"></canvas></div>
    </div>
    <div class="chart-panel">
      <div class="panel-title">GYROSCOPE (deg/s)</div>
      <div class="chart-wrap"><canvas id="gyroChart"></canvas></div>
    </div>
  </div>

  <div class="log-panel">
    <div class="panel-title">EPISODE LOG</div>
    <div class="log-scroll">
      <table>
        <thead><tr><th>TIME</th><th>EVENT</th><th>DURATION</th><th>SEVERITY</th></tr></thead>
        <tbody id="log-body"><tr><td colspan="4" style="text-align:center;padding:16px;font-family:var(--mono);font-size:10px;color:var(--text2);">No episodes recorded</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="sidebar">

    <div class="panel">
      <div class="panel-title">PATIENT</div>
      <div class="patient-card">
        <div class="patient-avatar">&#128100;</div>
        <div>
          <div class="patient-name-text" id="patient-name-display">Patient</div>
          <div class="patient-sub" id="mode-sub">WRIST &bull; EPILEPSY / FALL MODE</div>
        </div>
      </div>
      <div id="email-row" style="display:none;margin-bottom:8px;padding:7px 10px;background:var(--surface2);border-radius:8px;font-family:var(--mono);font-size:9px;color:var(--text2);">
        &#128231; <span id="caregiver-email-display" style="color:var(--accent)"></span>
      </div>
      <div class="buf-label"><span>DATA BUFFER</span><span id="buf-pct">0%</span></div>
      <div class="buf-track"><div class="buf-fill" id="buf-fill"></div></div>
      <div class="sample-count">SAMPLES: <span id="sample-count" style="color:var(--accent)">0</span></div>
    </div>

    <div class="panel">
      <div class="panel-title">STATISTICS</div>
      <div class="stats-grid">
        <div class="stat-box"><div class="stat-val" id="stat-total">0</div><div class="stat-lab">EPISODES</div></div>
        <div class="stat-box"><div class="stat-val" id="stat-longest">0s</div><div class="stat-lab">LONGEST</div></div>
        <div class="stat-box stat-full"><div class="stat-val" id="stat-freq" style="font-size:15px">N/A</div><div class="stat-lab">MOST FREQUENT</div></div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">CONTROLS</div>
      <div style="display:flex;flex-direction:column;gap:8px;">
        <button class="btn btn-primary" onclick="generateReport()" style="width:100%">&#128196; Generate PDF Report</button>
        <button class="btn btn-outline" onclick="clearLog()" style="width:100%">&#128465; Clear Episode Log</button>
      </div>
      <div style="margin-top:12px;padding:10px;background:var(--surface2);border-radius:8px;">
        <div style="font-family:var(--mono);font-size:8px;color:var(--text2);letter-spacing:2px;margin-bottom:6px;">DEVICE CONTROLS</div>
        <div style="font-size:11px;color:var(--text);line-height:1.8;font-family:var(--mono);">
          Short press = switch mode<br>
          Long press (2s) = test alert
        </div>
      </div>
    </div>

  </div>
</div>

<div id="notification"></div>

<script>
// Read URL params
const params  = new URLSearchParams(window.location.search);
const pname   = params.get('name')  || 'Patient';
const espIP   = params.get('ip')    || '';
const pemail  = params.get('email') || '';
const initMode = parseInt(params.get('mode') || '0');

document.getElementById('patient-name-display').textContent = pname;

// Tell server about patient name, email & initial mode
fetch('/set_patient', {
  method: 'POST',
  headers: {'Content-Type':'application/json'},
  body: JSON.stringify({name: pname, email: pemail, mode: initMode})
});

if (espIP) {
  document.getElementById('esp-ip-badge').style.display = 'inline-flex';
  document.getElementById('esp-ip-val').textContent = espIP;
}

if (pemail) {
  document.getElementById('email-row').style.display = 'block';
  document.getElementById('caregiver-email-display').textContent = pemail;
}

// Charts
const socket = io();
const POINTS = 250;

function makeChart(ctx, lbls, clrs) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array(POINTS).fill(''),
      datasets: lbls.map((l,i) => ({
        label: l, data: Array(POINTS).fill(null),
        borderColor: clrs[i], borderWidth: 1.5,
        pointRadius: 0, tension: 0.3, fill: false
      }))
    },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      scales: {
        x: {display:false},
        y: {
          beginAtZero:false,
          ticks: {color:'#6080a0', font:{family:'Share Tech Mono',size:9}},
          grid: {color:'rgba(26,42,74,0.5)'}
        }
      },
      plugins: {legend:{labels:{color:'#6080a0',font:{family:'Share Tech Mono',size:9},boxWidth:8}}}
    }
  });
}

const accelChart = makeChart(document.getElementById('accelChart').getContext('2d'),
  ['X','Y','Z'], ['#ff4466','#00cc66','#00aaff']);
const gyroChart  = makeChart(document.getElementById('gyroChart').getContext('2d'),
  ['X','Y','Z'], ['#ff9900','#cc44ff','#00ffcc']);

function pushChart(chart, vals) {
  chart.data.datasets.forEach((ds,i) => {
    ds.data.push(vals[i]);
    if (ds.data.length > POINTS) ds.data.shift();
  });
  chart.update('none');
}

let sampleCount = 0;
let bufCount    = 0;

socket.on('update_graphs', d => {
  pushChart(accelChart, [d.ax, d.ay, d.az]);
  pushChart(gyroChart,  [d.gx, d.gy, d.gz]);
  sampleCount++;
  bufCount = Math.min(bufCount + 1, 256);
  document.getElementById('sample-count').textContent = sampleCount;
  const pct = Math.round(bufCount / 256 * 100);
  document.getElementById('buf-fill').style.width  = pct + '%';
  document.getElementById('buf-pct').textContent  = pct + '%';
});

socket.on('update_activity', d => {
  const el    = document.getElementById('activity-display');
  const panel = document.getElementById('activity-panel');
  el.textContent = d.activity;
  el.style.color = d.color;
  el.style.textShadow = d.is_critical ? ('0 0 40px ' + d.color) : 'none';
  if (d.is_critical) {
    el.classList.add('flashing');
    panel.classList.add('critical-bg');
  } else {
    el.classList.remove('flashing');
    panel.classList.remove('critical-bg');
  }
  document.getElementById('conf-fill').style.width    = '92%';
  document.getElementById('conf-fill').style.background = d.color;
  document.getElementById('activity-time').textContent = new Date().toLocaleTimeString();
  bufCount = 0;
  setTimeout(() => { document.getElementById('conf-fill').style.width = '0%'; }, 2400);
});

socket.on('update_status', d => {
  const dot = document.getElementById('conn-dot');
  const txt = document.getElementById('conn-text');
  dot.className = d.esp_connected ? 'dot on' : 'dot';
  txt.textContent = d.esp_connected ? 'ESP32 Connected' : 'Waiting for device';
  updateModeUI(d.device_mode);
});

function updateModeUI(m) {
  const bw  = document.getElementById('btn-wrist');
  const ba  = document.getElementById('btn-ankle');
  const sub = document.getElementById('mode-sub');
  bw.className = 'mode-btn' + (m===0 ? ' active-wrist' : '');
  ba.className = 'mode-btn' + (m===1 ? ' active-ankle' : '');
  sub.textContent = m===0 ? 'WRIST \u2022 EPILEPSY / FALL MODE' : 'ANKLE \u2022 PARKINSON\u2019S / FOG MODE';
}

function switchMode(m) {
  fetch('/set_mode', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({mode: m})
  });
  updateModeUI(m);
}

socket.on('update_log', d => {
  const tbody = document.getElementById('log-body');
  if (!d.log || d.log.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:16px;font-family:var(--mono);font-size:10px;color:var(--text2);">No episodes recorded</td></tr>';
    return;
  }
  tbody.innerHTML = '';
  d.log.slice().reverse().forEach(ep => {
    const cls = ep.severity === 'CRITICAL' ? 'badge-critical' : 'badge-warning';
    const tr  = document.createElement('tr');
    tr.innerHTML = '<td>' + ep.start_time + '</td><td>' + ep.type + '</td><td>' + ep.duration_seconds + 's</td>'
      + '<td><span class="badge ' + cls + '">' + ep.severity + '</span></td>';
    tbody.appendChild(tr);
  });
});

socket.on('update_stats', d => {
  document.getElementById('stat-total').textContent  = d.total;
  document.getElementById('stat-longest').textContent = d.longest + 's';
  document.getElementById('stat-freq').textContent   = d.most_frequent;
});

socket.on('critical_alert', d => {
  const banner = document.getElementById('alert-banner');
  document.getElementById('alert-event').textContent = d.type + ' at ' + d.time;
  banner.style.display = 'block';
  setTimeout(() => { banner.style.display = 'none'; }, 12000);
});

let notifTimer = null;
socket.on('show_notification', d => { showNotif(d.message, d.type || 'info'); });
socket.on('connect', () => { showNotif('Connected to Sahwa server', 'success'); });

function showNotif(msg, type) {
  const el = document.getElementById('notification');
  el.textContent = msg;
  el.className   = 'notif-' + type;
  el.style.opacity = 1;
  if (notifTimer) clearTimeout(notifTimer);
  notifTimer = setTimeout(() => { el.style.opacity = 0; }, 3500);
}

function generateReport() { window.location.href = '/report'; }
function clearLog() {
  if (confirm('Clear all episode records?')) {
    fetch('/clear_log', {method:'POST'}).then(() => showNotif('Episode log cleared', 'info'));
  }
}

// Initial mode from URL
updateModeUI(initMode);
</script>
</body>
</html>"""

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/")
def login():
    return render_template_string(LOGIN_HTML)


@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/set_patient", methods=["POST"])
def set_patient():
    data = request.get_json(silent=True) or {}
    with data_lock:
        if "name" in data:
            state["patient_name"] = data["name"]
        if "email" in data and data["email"]:
            state["caregiver_email"] = data["email"]
            print("[Config] Caregiver email set to: " + data["email"])
        if "mode" in data:
            state["device_mode"] = int(data["mode"])
    return jsonify({"ok": True})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    data = request.get_json(silent=True) or {}
    m = int(data.get("mode", 0))
    with data_lock:
        state["device_mode"] = m
    mname = "Wrist" if m == 0 else "Ankle"
    print("[Web] Mode switched to " + mname)
    socketio.emit("update_status", {
        "esp_connected": state["esp_connected"],
        "device_mode": m,
        "ip": state["esp_ip"],
    })
    socketio.emit("show_notification", {
        "message": "Model switched to " + mname + " mode",
        "type": "info",
    })
    # Try to tell ESP32 too
    with data_lock:
        sock = esp_sock_ref[0]
    if sock:
        try:
            cmd = ("SET_MODE:" + str(m) + "\n").encode()
            sock.sendall(cmd)
        except Exception:
            pass
    return jsonify({"ok": True})


@app.route("/report")
def report():
    pname = state.get("patient_name", "Patient")
    pdf   = generate_pdf(pname)
    if pdf is None:
        return "PDF generation failed. Install reportlab.", 500
    fname = "Sahwa_Report_" + datetime.now().strftime("%Y%m%d_%H%M") + ".pdf"
    resp  = make_response(pdf.getvalue())
    resp.headers["Content-Type"]        = "application/pdf"
    resp.headers["Content-Disposition"] = "attachment; filename=" + fname
    return resp


@app.route("/clear_log", methods=["POST"])
def clear_log():
    global episode_log, current_episode, normal_streak
    episode_log     = []
    current_episode = None
    normal_streak   = 0
    if os.path.exists(EPISODES_FILE):
        os.remove(EPISODES_FILE)
    socketio.emit("update_log",   {"log": []})
    socketio.emit("update_stats", {"total": 0, "longest": 0, "most_frequent": "N/A"})
    return jsonify({"ok": True})



# ============================================================
# ESP32 WEBSOCKET NAMESPACE  (/esp32)
# The device connects here instead of raw TCP.
# Firmware: use ArduinoWebsockets, connect to
#   wss://YOUR-APP.up.railway.app/esp32
# ============================================================

@socketio.on("connect", namespace="/esp32")
def esp_connect():
    sid = request.sid
    ip  = request.remote_addr or "unknown"
    print("[WS/ESP32] Device connected: " + sid + " from " + ip)
    with data_lock:
        state["esp_connected"] = True
        state["esp_ip"]        = ip
        imu_buffer.clear()
    socketio.emit("update_status", {
        "esp_connected": True,
        "device_mode":   state["device_mode"],
        "ip":            ip,
    })
    socketio.emit("show_notification",
                  {"message": "ESP32 connected from " + ip, "type": "success"})


@socketio.on("disconnect", namespace="/esp32")
def esp_disconnect():
    print("[WS/ESP32] Device disconnected")
    with data_lock:
        state["esp_connected"] = False
        state["esp_ip"]        = ""
    socketio.emit("update_status",
                  {"esp_connected": False, "device_mode": state["device_mode"], "ip": ""})
    socketio.emit("show_notification",
                  {"message": "ESP32 disconnected", "type": "warning"})


@socketio.on("imu_data", namespace="/esp32")
def esp_imu(data):
    """Receive a single IMU row dict: {ax,ay,az,gx,gy,gz}"""
    try:
        row = {k: float(data[k]) for k in ("ax","ay","az","gx","gy","gz")}
        with data_lock:
            imu_buffer.append(row)
            state["samples"] += 1
        socketio.emit("update_graphs", row)
    except Exception as e:
        print("[WS/ESP32] imu_data error: " + str(e))


@socketio.on("imu_line", namespace="/esp32")
def esp_imu_line(data):
    """Receive CSV line: 'ts,ax,ay,az,gx,gy,gz' (same format as old TCP stream)"""
    try:
        line = str(data).strip()
        if line.startswith("MODE:"):
            m = int(line.split(":")[1])
            with data_lock:
                state["device_mode"] = m
            mname = "Wrist" if m == 0 else "Ankle"
            print("[WS/ESP32] Mode -> " + mname)
            socketio.emit("update_status",
                          {"esp_connected": True, "device_mode": m, "ip": state["esp_ip"]})
            socketio.emit("show_notification",
                          {"message": "Device switched to " + mname + " mode", "type": "info"})
            return
        if line == "TEST_ALERT":
            trigger_alert("TEST_ALERT")
            return
        parts = line.split(",")
        if len(parts) >= 7:
            row = {
                "ax": float(parts[1]), "ay": float(parts[2]), "az": float(parts[3]),
                "gx": float(parts[4]), "gy": float(parts[5]), "gz": float(parts[6]),
            }
            with data_lock:
                imu_buffer.append(row)
                state["samples"] += 1
            socketio.emit("update_graphs", row)
    except Exception as e:
        print("[WS/ESP32] imu_line error: " + str(e))


@socketio.on("mode_change", namespace="/esp32")
def esp_mode(data):
    m = int(data.get("mode", 0))
    with data_lock:
        state["device_mode"] = m
    mname = "Wrist" if m == 0 else "Ankle"
    socketio.emit("update_status",
                  {"esp_connected": True, "device_mode": m, "ip": state["esp_ip"]})
    socketio.emit("show_notification",
                  {"message": "Device switched to " + mname + " mode", "type": "info"})


@socketio.on("connect")
def on_connect():
    with data_lock:
        s = dict(state)
    socketio.emit("update_status", {
        "esp_connected": s["esp_connected"],
        "device_mode":   s["device_mode"],
        "ip":            s.get("esp_ip",""),
    })
    socketio.emit("update_activity", {
        "activity":    s["prediction"],
        "color":       s["color"],
        "is_critical": s["is_critical"],
        "label":       s["pred_label"],
    })
    socketio.emit("update_log",   {"log": get_log_json()})
    socketio.emit("update_stats", get_stats())

# ============================================================
# MAIN
# ============================================================
def load_episodes_from_disk():
    global episode_log
    if os.path.exists(EPISODES_FILE):
        try:
            with open(EPISODES_FILE) as f:
                episode_log = json.load(f)
            print("[Main] Loaded " + str(len(episode_log)) + " previous episodes")
        except Exception:
            episode_log = []


def main():
    print("=" * 60)
    print("  Sahwa Monitoring System v2.0")
    print("  Abu Dhabi University — URIC 2026")
    print("=" * 60)
    load_episodes_from_disk()
    models = load_models()
    # TCPServer removed — ESP32 now connects via WebSocket
    InferenceEngine(models).start()
    url = "http://127.0.0.1:" + str(WEB_PORT)
    print("[Web] Dashboard: " + url)
    print("[Info] Set env vars: GMAIL_SENDER, GMAIL_APP_PASSWORD")
    print("=" * 60)
    # Auto-open disabled for cloud deployment
    port = int(os.environ.get("PORT", WEB_PORT))
    print("[Web] Running on port " + str(port))
    print("[WS]  ESP32 connects to wss://YOUR-APP.up.railway.app/esp32")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
