# -*- coding: utf-8 -*-
"""
sahwa Server v4.0 — Full Stack, Flask only, SSE real-time
No SocketIO required. Uses Server-Sent Events for real-time push.
"""

import os, sys, json, socket, threading, time, webbrowser, smtplib, io, queue
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import joblib
from flask import Flask, Response, render_template_string, make_response, jsonify, request, stream_with_context

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════
TCP_PORT   = 8888
WEB_PORT   = 5000
WIN        = 256
SR         = 50
SMOOTH     = 3
EP_FILE    = "episodes.json"

GMAIL_SENDER       = "sahwa.alerts@gmail.com"

# XGBoost availability — install with: pip install xgboost
try:
    import xgboost  # noqa
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
GMAIL_APP_PASSWORD = "your-app-password"

WLABELS = {0:"Stand",1:"Walk",2:"FastWalk",3:"Sit",4:"SitStand",6:"Fall",8:"Seizure"}
ALABELS = {0:"Stand",1:"Walk",2:"FastWalk",3:"Sit",4:"SitStand",5:"Stairs",7:"FoG"}
CRIT_W  = {6, 8}
CRIT_A  = {7}
WARN    = {"FOG","SITSTAND"}

# ═══════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════
lock         = threading.Lock()
buf          = deque()
esp_ref      = [None]
clients      = []          # SSE subscriber queues
client_lock  = threading.Lock()

state = {
    "mode":0, "connected":False,
    "activity":"Waiting for device...", "label":-1,
    "color":"#0ea5e9", "critical":False,
    "samples":0, "patient":"Patient",
    "esp_ip":"", "caregiver_email":"",
}
episodes       = []
cur_ep         = None
norm_streak    = 0

# ═══════════════════════════════════════════════════════════════════════════
# SSE BROADCAST
# ═══════════════════════════════════════════════════════════════════════════
def broadcast(event, data):
    msg = "event: {}\ndata: {}\n\n".format(event, json.dumps(data))
    dead = []
    with client_lock:
        for q in clients:
            try: q.put_nowait(msg)
            except Exception: dead.append(q)
        for q in dead: clients.remove(q)

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
def spectral_entropy(psd):
    psd = psd[psd > 0]
    if len(psd) == 0: return 0.0
    psd = psd / psd.sum()
    return float(-np.sum(psd * np.log2(psd)) / np.log2(len(psd)))

def extract(df):
    df = df.copy()
    df["am"] = np.sqrt(df.ax**2 + df.ay**2 + df.az**2)
    df["gm"] = np.sqrt(df.gx**2 + df.gy**2 + df.gz**2)
    feats = []
    for sig in ["ax","ay","az","gx","gy","gz","am","gm"]:
        d = df[sig].values.astype(float)
        feats += [float(np.mean(d)), float(np.std(d)),
                  float(np.min(d)),  float(np.max(d)),
                  float(np.sqrt(np.mean(d**2)))]
        c = d - np.mean(d)
        feats.append(float(len(np.where(np.diff(np.sign(c)))[0]) / len(d)))
        yf = fft(d); n = len(d)
        mags = np.abs(yf[1:n//2])
        freqs = fftfreq(n, 1.0/SR)[1:n//2]
        if len(mags) > 0:
            feats.append(float(freqs[np.argmax(mags)]))
            feats.append(spectral_entropy(mags**2))
        else:
            feats += [0.0, 0.0]
    return np.array(feats).reshape(1, -1)

# ═══════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════
class Pack:
    def __init__(self, mp, sp, jp, crit, lbls):
        self.model  = joblib.load(mp)
        self.scaler = joblib.load(sp)
        with open(jp) as f: raw = json.load(f)
        self.mapping = {int(k): v for k,v in raw.items()}
        self.crit = crit; self.lbls = lbls
    def predict(self, df):
        idx  = int(self.model.predict(self.scaler.transform(extract(df)))[0])
        orig = next((k for k,v in self.mapping.items() if v == idx), -1)
        return orig, self.lbls.get(orig, "Unknown"), orig in self.crit

def load_models():
    models = {}

    # WRIST: XGBoost (96.88%) preferred, RF fallback (94.62%)
    wxgb = ["wrist_xgb_model.pkl","wrist_xgb_scaler.pkl","wrist_label_mapping.json"]
    wrf  = ["wrist_rf_model.pkl","wrist_rf_scaler.pkl","wrist_label_mapping.json"]
    if HAS_XGB and all(os.path.exists(f) for f in wxgb):
        models[0] = Pack(*wxgb, CRIT_W, WLABELS)
        print("[Models] Wrist XGBoost loaded (96.88% accuracy)")
    elif all(os.path.exists(f) for f in wrf):
        models[0] = Pack(*wrf, CRIT_W, WLABELS)
        print("[Models] Wrist Random Forest loaded (94.62% accuracy)")
        print("[Models]   Run: pip install xgboost  to unlock 96.88% model")
    else:
        print("[Models] WARNING: No wrist model files found")

    # ANKLE: Random Forest (97.57%) is the best model
    afiles = ["ankle_rf_model.pkl","ankle_rf_scaler.pkl","ankle_label_mapping.json"]
    if all(os.path.exists(f) for f in afiles):
        models[1] = Pack(*afiles, CRIT_A, ALABELS)
        print("[Models] Ankle Random Forest loaded (97.57% accuracy)")
    else:
        print("[Models] WARNING: No ankle model files found")

    if not models:
        print("[Models] ERROR: No model files found in current directory")
        sys.exit(1)
    return models

# ═══════════════════════════════════════════════════════════════════════════
# EPISODES
# ═══════════════════════════════════════════════════════════════════════════
def ep_start(t):
    global cur_ep
    cur_ep = {"type":t, "start":datetime.now().isoformat(),
               "end":None, "dur":0,
               "severity":"CRITICAL" if t in ["SEIZURE","FALL"] else "WARNING"}

def ep_end():
    global cur_ep, episodes
    if not cur_ep: return
    cur_ep["end"] = datetime.now().isoformat()
    s = datetime.fromisoformat(cur_ep["start"])
    e = datetime.fromisoformat(cur_ep["end"])
    cur_ep["dur"] = int((e - s).total_seconds())
    episodes.append(cur_ep)
    try:
        with open(EP_FILE,"w") as f: json.dump(episodes, f, indent=2)
    except Exception: pass
    broadcast("log",   {"log": ep_log()})
    broadcast("stats", ep_stats())
    cur_ep = None

def ep_log():
    return [{"time": e["start"][:19].replace("T"," "),
             "type": e["type"], "dur": e["dur"],
             "severity": e["severity"]} for e in episodes[-30:]]

def ep_stats():
    if not episodes: return {"total":0,"longest":0,"freq":"N/A"}
    types = [e["type"] for e in episodes]
    return {"total":len(episodes),
            "longest":max(e["dur"] for e in episodes),
            "freq":max(set(types), key=types.count)}

def load_episodes():
    global episodes
    if os.path.exists(EP_FILE):
        try:
            with open(EP_FILE) as f: episodes = json.load(f)
            print("[Main] Loaded {} previous episodes".format(len(episodes)))
        except Exception: episodes = []

# ═══════════════════════════════════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════════════════════════════════
def alert(event_type):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[ALERT] {} at {}".format(event_type, now))
    broadcast("alert", {"type":event_type, "time":now})
    broadcast("notif", {"msg":"CRITICAL: {} detected at {}".format(event_type, now), "t":"critical"})
    with lock:
        sock = esp_ref[0]
    if sock:
        try: sock.sendall(b"ALERT\n")
        except Exception: pass
    threading.Thread(target=_email, args=(event_type, now), daemon=True).start()

def _email(event_type, ts):
    to = state.get("caregiver_email","").strip()
    if not to or GMAIL_APP_PASSWORD == "your-app-password": return
    try:
        msg = MIMEMultipart()
        msg["From"] = GMAIL_SENDER; msg["To"] = to
        msg["Subject"] = "sahwa ALERT: " + event_type
        msg.attach(MIMEText(
            "Patient: {}\nEvent: {}\nTime: {}\n\nPlease check immediately.\n-- sahwa".format(
                state.get("patient","Patient"), event_type, ts), "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            s.sendmail(GMAIL_SENDER, to, msg.as_string())
        print("[Email] Sent to " + to)
    except Exception as e:
        print("[Email] Failed: " + str(e))

# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
class Inference(threading.Thread):
    def __init__(self, models):
        super().__init__(daemon=True)
        self.models = models; self.hist = []
    def run(self):
        global norm_streak, cur_ep
        while True:
            with lock: n = len(buf)
            if n >= WIN:
                with lock:
                    window = [buf.popleft() for _ in range(WIN)]
                try:
                    df   = pd.DataFrame(window)
                    mode = state["mode"]
                    pack = self.models.get(mode)
                    if pack is None: time.sleep(0.1); continue
                    orig, name, crit_raw = pack.predict(df)
                    self.hist.append(orig)
                    if len(self.hist) > SMOOTH: self.hist.pop(0)
                    confirmed = (len(self.hist) == SMOOTH and
                                 all(p == orig for p in self.hist))
                    crit  = crit_raw and confirmed
                    name_up = name.upper()
                    color = "#ef4444" if crit else ("#f97316" if name_up in WARN else "#22c55e")
                    with lock:
                        state.update({"activity":name_up,"label":orig,
                                      "color":color,"critical":crit})
                    broadcast("activity", {"activity":name_up,"color":color,"critical":crit})
                    if crit:
                        norm_streak = 0
                        if cur_ep is None:
                            ep_start(name_up); alert(name_up)
                    else:
                        if cur_ep is not None:
                            norm_streak += 1
                            if norm_streak >= 3: ep_end(); norm_streak = 0
                        else: norm_streak = 0
                except Exception as e:
                    print("[Inference] " + str(e))
            else:
                time.sleep(0.05)

# ═══════════════════════════════════════════════════════════════════════════
# TCP SERVER
# ═══════════════════════════════════════════════════════════════════════════
class TCPServer(threading.Thread):
    def __init__(self, models):
        super().__init__(daemon=True)
        self.models = models
    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", TCP_PORT)); srv.listen(1); srv.settimeout(1.0)
        print("[TCP] Listening on :{}".format(TCP_PORT))
        while True:
            try:
                conn, addr = srv.accept()
                ip = str(addr[0])
                print("[TCP] ESP32 connected from " + ip)
                with lock:
                    esp_ref[0] = conn; state["connected"] = True; state["esp_ip"] = ip
                    buf.clear()
                broadcast("status", {"connected":True,"mode":state["mode"],"ip":ip})
                broadcast("notif",  {"msg":"ESP32 connected from "+ip,"t":"success"})
                self._client(conn)
                with lock:
                    esp_ref[0] = None; state["connected"] = False; state["esp_ip"] = ""
                print("[TCP] Disconnected")
                broadcast("status", {"connected":False,"mode":state["mode"],"ip":""})
                broadcast("notif",  {"msg":"ESP32 disconnected","t":"warning"})
            except socket.timeout: continue
            except Exception as e: print("[TCP] "+str(e)); time.sleep(1)

    def _client(self, conn):
        b = ""; conn.settimeout(5.0)
        try:
            while True:
                try: data = conn.recv(1024)
                except socket.timeout: continue
                if not data: break
                b += data.decode("utf-8", errors="ignore")
                while "\n" in b:
                    line, b = b.split("\n", 1)
                    line = line.strip()
                    if not line: continue
                    if line.startswith("MODE:"):
                        try:
                            m = int(line.split(":")[1])
                            with lock: state["mode"] = m
                            mn = "Wrist" if m == 0 else "Ankle"
                            broadcast("status", {"connected":True,"mode":m,"ip":state["esp_ip"]})
                            broadcast("notif",  {"msg":"Switched to "+mn+" mode","t":"info"})
                        except Exception: pass
                    elif line == "TEST_ALERT":
                        alert("TEST_ALERT")
                    else:
                        parts = line.split(",")
                        if len(parts) >= 7:
                            try:
                                row = {"ax":float(parts[1]),"ay":float(parts[2]),"az":float(parts[3]),
                                       "gx":float(parts[4]),"gy":float(parts[5]),"gz":float(parts[6])}
                                with lock: buf.append(row); state["samples"] += 1
                                broadcast("imu", row)
                            except Exception: pass
        except Exception as e:
            print("[TCP] client: " + str(e))

# ═══════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════
app = Flask(__name__)

@app.route("/events")
def events():
    def stream():
        q = queue.Queue(maxsize=200)
        with client_lock: clients.append(q)
        # Send current state immediately
        yield "event: status\ndata: {}\n\n".format(
            json.dumps({"connected":state["connected"],"mode":state["mode"],"ip":state["esp_ip"]}))
        yield "event: activity\ndata: {}\n\n".format(
            json.dumps({"activity":state["activity"],"color":state["color"],"critical":state["critical"]}))
        yield "event: log\ndata: {}\n\n".format(json.dumps({"log":ep_log()}))
        yield "event: stats\ndata: {}\n\n".format(json.dumps(ep_stats()))
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield msg
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            with client_lock:
                if q in clients: clients.remove(q)
    return Response(stream_with_context(stream()),
                    content_type="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/set_patient", methods=["POST"])
def set_patient():
    d = request.get_json(silent=True) or {}
    with lock:
        if "name"  in d: state["patient"] = d["name"]
        if "email" in d: state["caregiver_email"] = d["email"]
        if "mode"  in d: state["mode"] = int(d["mode"])
    if d.get("email"):
        print("[Config] Caregiver email: " + d["email"])
    return jsonify({"ok":True})

@app.route("/api/set_mode", methods=["POST"])
def set_mode():
    d = request.get_json(silent=True) or {}
    m = int(d.get("mode", 0))
    with lock: state["mode"] = m
    mn = "Wrist" if m == 0 else "Ankle"
    broadcast("status", {"connected":state["connected"],"mode":m,"ip":state["esp_ip"]})
    broadcast("notif",  {"msg":"Switched to "+mn+" mode","t":"info"})
    with lock:
        sock = esp_ref[0]
    if sock:
        try: sock.sendall(("SET_MODE:{}\n".format(m)).encode())
        except Exception: pass
    return jsonify({"ok":True})

@app.route("/api/clear_log", methods=["POST"])
def clear_log():
    global episodes, cur_ep, norm_streak
    episodes = []; cur_ep = None; norm_streak = 0
    if os.path.exists(EP_FILE): os.remove(EP_FILE)
    broadcast("log",   {"log":[]})
    broadcast("stats", {"total":0,"longest":0,"freq":"N/A"})
    return jsonify({"ok":True})

@app.route("/api/report")
def report():
    pname = state.get("patient","Patient")
    pdf   = _gen_pdf(pname)
    if pdf is None: return "PDF generation failed. Install reportlab.", 500
    fname = "sahwa_{}.pdf".format(datetime.now().strftime("%Y%m%d_%H%M"))
    resp  = make_response(pdf.getvalue())
    resp.headers["Content-Type"]        = "application/pdf"
    resp.headers["Content-Disposition"] = "attachment; filename=" + fname
    return resp

@app.route("/api/state")
def api_state():
    return jsonify({
        "connected": state["connected"],
        "mode":      state["mode"],
        "activity":  state["activity"],
        "color":     state["color"],
        "critical":  state["critical"],
        "samples":   state["samples"],
        "patient":   state["patient"],
        "ip":        state["esp_ip"],
        "log":       ep_log(),
        "stats":     ep_stats(),
    })

def _gen_pdf(patient_name):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        buf2 = io.BytesIO()
        doc  = SimpleDocTemplate(buf2, pagesize=letter,
                                  topMargin=.75*inch, bottomMargin=.75*inch,
                                  leftMargin=inch, rightMargin=inch)
        st   = getSampleStyleSheet()
        els  = [Paragraph("sahwa Report", st["Title"]),
                Spacer(1,8),
                Paragraph("Patient: "+patient_name, st["Normal"]),
                Paragraph("Generated: "+datetime.now().strftime("%Y-%m-%d %H:%M:%S"), st["Normal"]),
                Spacer(1,16)]
        stats = ep_stats()
        tbl = Table([["Metric","Value"],["Total Episodes",str(stats["total"])],
                     ["Longest",str(stats["longest"])+"s"],["Most Frequent",stats["freq"]]],
                    colWidths=[3*inch,3*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0f172a")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),.5,colors.grey),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafc")]),
        ]))
        els += [tbl, Spacer(1,16)]
        if episodes:
            rows = [["Time","Event","Duration","Severity"]]
            for ep in reversed(episodes):
                rows.append([ep["start"][:19].replace("T"," "),
                              ep["type"], str(ep["dur"])+"s", ep["severity"]])
            t2 = Table(rows, colWidths=[2*inch,1.5*inch,1*inch,1.5*inch])
            t2.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0f172a")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("GRID",(0,0),(-1,-1),.5,colors.grey),
            ]))
            els.append(t2)
        else:
            els.append(Paragraph("No episodes recorded.", st["Normal"]))
        doc.build(els)
        buf2.seek(0); return buf2
    except Exception as e:
        print("[PDF] "+str(e)); return None

# ═══════════════════════════════════════════════════════════════════════════
# HTML PAGES
# ═══════════════════════════════════════════════════════════════════════════

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sahwa — Setup</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#030609;--s1:#070d17;--s2:#0a1220;--b1:#16263d;--b2:#1e3252;
--acc:#0ea5e9;--a2:#06d6a0;--t1:#e2eaf4;--t2:#7ea3c4;--t3:#405d7a;
--sans:'Inter',sans-serif;--mono:'JetBrains Mono',monospace}
html,body{height:100%;background:var(--bg);font-family:var(--sans);color:var(--t1);overflow:hidden}
canvas{position:fixed;inset:0;z-index:0;pointer-events:none}
.wrap{position:relative;z-index:1;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px}
.card{background:rgba(7,13,23,0.92);border:1px solid var(--b1);border-radius:24px;padding:44px 40px;width:440px;backdrop-filter:blur(24px);animation:ci .8s cubic-bezier(.16,1,.3,1) both}
@keyframes ci{from{opacity:0;transform:translateY(28px) scale(.97)}to{opacity:1;transform:none}}
.logo{display:flex;align-items:center;gap:12px;margin-bottom:28px}
.licon{width:46px;height:46px;background:linear-gradient(135deg,var(--acc),var(--a2));border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;animation:hb 2s ease-in-out infinite;flex-shrink:0}
@keyframes hb{0%,100%{transform:scale(1)}14%{transform:scale(1.1)}28%{transform:scale(1)}42%{transform:scale(1.06)}}
.ltxt{font-size:24px;font-weight:800;letter-spacing:2.5px;background:linear-gradient(90deg,var(--acc),var(--a2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.lsub{font-size:9px;color:var(--t3);letter-spacing:2px;font-family:var(--mono);margin-top:2px}
.sep{border-bottom:1px solid var(--b1);margin-bottom:24px;padding-bottom:20px}
.field{margin-bottom:18px}
.flabel{font-size:10px;font-family:var(--mono);color:var(--t3);letter-spacing:2px;margin-bottom:7px;display:flex;align-items:center;gap:5px}
.req{color:var(--acc)}
input[type=text],input[type=email]{width:100%;background:var(--s2);border:1px solid var(--b1);border-radius:10px;padding:12px 14px;font-family:var(--mono);font-size:13.5px;color:var(--t1);outline:none;transition:border-color .25s,box-shadow .25s}
input:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(14,165,233,.12)}
input::placeholder{color:var(--t3)}
.hint{background:rgba(14,165,233,.07);border:1px solid rgba(14,165,233,.15);border-radius:8px;padding:9px 13px;font-size:11.5px;color:var(--t2);margin-top:7px;line-height:1.55}
.hint strong{color:var(--acc)}
.mode-wrap{margin-bottom:18px}
.mlabel{font-size:10px;font-family:var(--mono);color:var(--t3);letter-spacing:2px;margin-bottom:9px}
.mopts{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.mopt{background:var(--s2);border:2px solid var(--b1);border-radius:12px;padding:13px;cursor:pointer;transition:all .22s;text-align:center;user-select:none}
.mopt:hover{border-color:rgba(14,165,233,.4)}
.mopt.sel-w{border-color:var(--acc);background:rgba(14,165,233,.08)}
.mopt.sel-a{border-color:#a78bfa;background:rgba(167,139,250,.08)}
.micon{font-size:20px;margin-bottom:5px}
.mname{font-weight:700;font-size:13px;letter-spacing:.5px}
.mdesc{font-family:var(--mono);font-size:9px;color:var(--t3);margin-top:2px}
.btn{width:100%;padding:14px;background:linear-gradient(135deg,var(--acc),#0284c7);border:none;border-radius:11px;color:#fff;font-size:15px;font-weight:700;cursor:pointer;transition:all .25s;letter-spacing:.5px;margin-top:4px}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(14,165,233,.45)}
.btn:active{transform:translateY(0)}
.ver{text-align:center;margin-top:18px;font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:1.5px}
</style></head><body>
<canvas id="c"></canvas>
<div class="wrap"><div class="card">
  <div class="logo sep">
    <div class="licon">&#10084;</div>
    <div><div class="ltxt">sahwa</div><div class="lsub">NEUROLOGICAL MONITORING SYSTEM</div></div>
  </div>
  <div class="field">
    <div class="flabel">PATIENT NAME <span class="req">*</span></div>
    <input type="text" id="pname" placeholder="Enter patient name" autocomplete="off">
  </div>
  <div class="field">
    <div class="flabel">CAREGIVER EMAIL <span style="color:var(--t3);font-size:9px">(optional)</span></div>
    <input type="email" id="email" placeholder="caregiver@example.com" autocomplete="off">
  </div>
  <div class="field">
    <div class="flabel">ESP32 IP ADDRESS <span class="req">*</span></div>
    <input type="text" id="ip" placeholder="e.g. 192.168.137.xxx" autocomplete="off">
    <div class="hint">Power on device &rarr; check OLED display for IP.<br>Laptop hotspot must be active: <strong>ysfthecreator</strong><br>For demo (no hardware): type <strong>localhost</strong></div>
  </div>
  <div class="mode-wrap">
    <div class="mlabel">DETECTION MODE</div>
    <div class="mopts">
      <div class="mopt sel-w" id="mw" onclick="selMode(0)">
        <div class="micon">&#9785;</div>
        <div class="mname" style="color:#38bdf8">WRIST</div>
        <div class="mdesc">SEIZURE + FALL</div>
      </div>
      <div class="mopt" id="ma" onclick="selMode(1)">
        <div class="micon">&#128694;</div>
        <div class="mname" style="color:#a78bfa">ANKLE</div>
        <div class="mdesc">FOG + GAIT</div>
      </div>
    </div>
  </div>
  <button class="btn" onclick="go()">&#9654;&nbsp;&nbsp;START MONITORING</button>
  <div class="ver">sahwa v4.0 &bull; Abu Dhabi University URIC 2026</div>
</div></div>
<script>
let mode=0;
function selMode(m){mode=m;document.getElementById('mw').className='mopt'+(m===0?' sel-w':'');document.getElementById('ma').className='mopt'+(m===1?' sel-a':'')}
function go(){
  const n=document.getElementById('pname').value.trim();
  const e=document.getElementById('email').value.trim();
  const ip=document.getElementById('ip').value.trim();
  if(!n){document.getElementById('pname').focus();return}
  if(!ip){document.getElementById('ip').focus();return}
  fetch('/api/set_patient',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:n,email:e,mode:mode})});
  window.location.href='/dashboard?name='+encodeURIComponent(n)+'&email='+encodeURIComponent(e)+'&ip='+encodeURIComponent(ip)+'&mode='+mode;
}
document.addEventListener('keydown',e=>{if(e.key==='Enter')go()});
// Particle canvas
const cv=document.getElementById('c'),cx=cv.getContext('2d');
let W,H,pts=[];
function rsz(){W=cv.width=innerWidth;H=cv.height=innerHeight}rsz();addEventListener('resize',rsz);
for(let i=0;i<60;i++)pts.push({x:Math.random()*1920,y:Math.random()*1080,vx:(Math.random()-.5)*.2,vy:(Math.random()-.5)*.2,r:Math.random()*1.2+.4,a:Math.random()*.3+.07});
function draw(){cx.clearRect(0,0,W,H);pts.forEach(p=>{p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;cx.beginPath();cx.arc(p.x,p.y,p.r,0,Math.PI*2);cx.fillStyle='rgba(14,165,233,'+p.a+')';cx.fill()});for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<110){cx.beginPath();cx.moveTo(pts[i].x,pts[i].y);cx.lineTo(pts[j].x,pts[j].y);cx.strokeStyle='rgba(14,165,233,'+(0.06*(1-d/110))+')';cx.lineWidth=.5;cx.stroke()}}requestAnimationFrame(draw)}draw();
</script></body></html>"""


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sahwa Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#030609;--s1:#070d17;--s2:#0a1220;--s3:#0e1929;
--b1:#16263d;--b2:#1e3252;--acc:#0ea5e9;--a2:#06d6a0;
--green:#22c55e;--orange:#f97316;--red:#ef4444;
--t1:#e2eaf4;--t2:#7ea3c4;--t3:#405d7a;
--sans:'Inter',sans-serif;--mono:'JetBrains Mono',monospace}
html,body{height:100%;background:var(--bg);color:var(--t1);font-family:var(--sans);overflow:hidden}

/* ALERT BANNER */
#banner{display:none;position:fixed;top:0;left:0;right:0;z-index:200;background:var(--red);color:#fff;text-align:center;padding:12px;font-weight:700;font-size:15px;letter-spacing:2px;animation:fp .5s infinite alternate}
@keyframes fp{from{opacity:1}to{opacity:.45}}
#banner.on{display:block}

/* HEADER */
header{height:58px;display:flex;align-items:center;justify-content:space-between;padding:0 24px;background:rgba(3,6,9,.95);border-bottom:1px solid var(--b1);flex-shrink:0;position:relative;z-index:100}
.hlogo{display:flex;align-items:center;gap:10px;text-decoration:none}
.hicon{width:30px;height:30px;background:linear-gradient(135deg,var(--acc),var(--a2));border-radius:7px;display:flex;align-items:center;justify-content:center;font-size:14px;animation:hb 2s ease-in-out infinite}
@keyframes hb{0%,100%{transform:scale(1)}14%{transform:scale(1.1)}28%{transform:scale(1)}42%{transform:scale(1.06)}}
.htext{font-size:16px;font-weight:800;letter-spacing:2px;background:linear-gradient(90deg,var(--acc),var(--a2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hmid{display:flex;align-items:center;gap:10px}
.pill{display:flex;align-items:center;gap:6px;padding:5px 12px;border:1px solid var(--b1);border-radius:20px;font-family:var(--mono);font-size:11px;background:var(--s2)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--t3)}
.dot.on{background:var(--green);box-shadow:0 0 7px var(--green);animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.mode-toggle{display:flex;background:var(--s2);border:1px solid var(--b1);border-radius:8px;overflow:hidden}
.mtb{padding:6px 14px;font-family:var(--mono);font-size:11px;letter-spacing:1px;cursor:pointer;border:none;background:transparent;color:var(--t3);transition:all .22s}
.mtb:hover{color:var(--t1)}
.mtb.aw{background:rgba(14,165,233,.15);color:#38bdf8}
.mtb.aa{background:rgba(167,139,250,.15);color:#a78bfa}
.hright{display:flex;align-items:center;gap:8px}
.hbtn{padding:6px 14px;border-radius:7px;font-size:12px;font-weight:600;cursor:pointer;transition:all .2s;border:none;letter-spacing:.5px}
.hbtn-p{background:linear-gradient(135deg,var(--acc),#0284c7);color:#fff}
.hbtn-p:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(14,165,233,.4)}
.hbtn-o{background:transparent;border:1px solid var(--b1) !important;border-style:solid;color:var(--t2)}
.hbtn-o:hover{border-color:var(--acc) !important;color:var(--acc)}
.ipbadge{font-family:var(--mono);font-size:10px;color:var(--t3);background:var(--s2);border:1px solid var(--b1);padding:3px 9px;border-radius:5px;display:none}

/* LAYOUT */
.main{height:calc(100vh - 58px);display:grid;grid-template-columns:1fr 284px;grid-template-rows:auto 1fr auto;gap:0;overflow:hidden}

/* ACTIVITY PANEL */
.apanel{grid-column:1;background:var(--s1);border-bottom:1px solid var(--b1);border-right:1px solid var(--b1);display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;min-height:150px;position:relative;overflow:hidden;transition:background .5s}
.apanel.crit{background:rgba(239,68,68,.06)}
.albl{font-family:var(--mono);font-size:9.5px;color:var(--t3);letter-spacing:3px;margin-bottom:10px}
#aword{font-size:50px;font-weight:900;letter-spacing:4px;transition:color .4s,text-shadow .4s;text-align:center}
#aword.crit{animation:af .4s infinite alternate}
@keyframes af{from{opacity:1}to{opacity:.2}}
.cbar-wrap{width:65%;height:3px;background:var(--b1);border-radius:2px;margin-top:10px;overflow:hidden}
.cbar-fill{height:100%;width:0;border-radius:2px;transition:width .6s,background .4s}
.atime{font-family:var(--mono);font-size:10px;color:var(--t3);margin-top:7px}

/* CHARTS */
.charts{grid-column:1;display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--b1);border-right:1px solid var(--b1)}
.cpanel{padding:14px 16px;border-right:1px solid var(--b1);position:relative}
.cpanel:last-child{border-right:none}
.ctitle{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:2.5px;margin-bottom:10px}
.ccanvas{height:90px;position:relative}

/* LOG */
.logpanel{grid-column:1;border-right:1px solid var(--b1);display:flex;flex-direction:column;overflow:hidden;max-height:200px}
.lphead{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid var(--b1);flex-shrink:0}
.lphead .ctitle{margin:0}
.lpbtn{background:transparent;border:1px solid var(--b1);border-radius:5px;color:var(--t3);font-size:10px;padding:2px 8px;cursor:pointer;font-family:var(--mono);transition:all .2s}
.lpbtn:hover{border-color:var(--red);color:var(--red)}
.logscroll{overflow-y:auto;flex:1}
.logscroll::-webkit-scrollbar{width:3px}
.logscroll::-webkit-scrollbar-thumb{background:var(--b1)}
table{width:100%;border-collapse:collapse;font-size:11px}
thead th{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:2px;padding:6px 10px;text-align:left;border-bottom:1px solid rgba(22,38,61,.6);position:sticky;top:0;background:var(--s1)}
tbody tr{border-bottom:1px solid rgba(22,38,61,.4);transition:background .15s}
tbody tr:hover{background:var(--s2)}
tbody td{padding:6px 10px;font-family:var(--mono);font-size:10px}
.badge{padding:2px 7px;border-radius:3px;font-size:8.5px;font-weight:700}
.bc{background:rgba(239,68,68,.18);color:var(--red)}
.bw{background:rgba(249,115,22,.18);color:var(--orange)}

/* SIDEBAR */
.sidebar{grid-column:2;grid-row:1/4;display:flex;flex-direction:column;overflow-y:auto;background:var(--s1)}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:var(--b1)}
.spanel{border-bottom:1px solid var(--b1);padding:16px}
.shead{font-family:var(--mono);font-size:9px;color:var(--t3);letter-spacing:2.5px;margin-bottom:12px;display:flex;align-items:center;gap:6px}
.shead::after{content:'';flex:1;height:1px;background:var(--b1)}
.patcard{display:flex;align-items:center;gap:11px;padding:11px;background:var(--s2);border-radius:10px;margin-bottom:10px}
.avatar{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,var(--acc),var(--a2));display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0}
.pname{font-weight:700;font-size:14px}
.psub{font-family:var(--mono);font-size:8.5px;color:var(--t3);margin-top:2px}
.emlrow{margin-bottom:10px;padding:7px 10px;background:var(--s2);border-radius:8px;font-family:var(--mono);font-size:9px;color:var(--t3);display:none}
.buflbl{display:flex;justify-content:space-between;font-family:var(--mono);font-size:9px;color:var(--t3);margin-bottom:4px}
.buftrack{height:3px;background:var(--b1);border-radius:2px;overflow:hidden}
.buffill{height:100%;width:0;background:linear-gradient(90deg,var(--acc),var(--a2));border-radius:2px;transition:width .3s}
.samplbl{font-family:var(--mono);font-size:9px;color:var(--t3);margin-top:5px}
.sgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.sbox{background:var(--s2);border-radius:9px;padding:10px;text-align:center}
.sval{font-family:var(--mono);font-size:19px;font-weight:700;color:var(--acc)}
.slbl{font-size:9px;color:var(--t3);letter-spacing:1px;margin-top:3px}
.sfull{grid-column:1/-1}
.cbtn{width:100%;padding:11px;border:none;border-radius:9px;font-size:13px;font-weight:600;cursor:pointer;transition:all .2s;letter-spacing:.5px;margin-bottom:8px}
.cbtn-p{background:linear-gradient(135deg,var(--acc),#0284c7);color:#fff}
.cbtn-p:hover{transform:translateY(-1px);box-shadow:0 4px 18px rgba(14,165,233,.4)}
.cbtn-o{background:transparent;border:1px solid var(--b1) !important;border-style:solid;color:var(--t2)}
.cbtn-o:hover{border-color:var(--red) !important;color:var(--red)}
.devnotes{background:var(--s2);border-radius:8px;padding:10px}
.devnotes p{font-family:var(--mono);font-size:9.5px;color:var(--t2);line-height:1.8}

/* NOTIFICATION */
#notif{position:fixed;bottom:20px;right:20px;padding:10px 15px;border-radius:9px;font-family:var(--mono);font-size:11px;opacity:0;transition:opacity .3s;max-width:280px;z-index:300;border-left:3px solid transparent;background:var(--s2);border-color:var(--b2)}
#notif.on{opacity:1}
#notif.nc{border-left-color:var(--red)}
#notif.ns{border-left-color:var(--green)}
#notif.nw{border-left-color:var(--orange)}
#notif.ni{border-left-color:var(--acc)}

@media(max-width:900px){.charts{grid-template-columns:1fr}.main{grid-template-columns:1fr}}
</style></head>
<body>
<div id="banner">&#9888; CRITICAL: <span id="btext"></span> DETECTED &#9888;</div>

<header>
  <a href="/" class="hlogo"><div class="hicon">&#10084;</div><span class="htext">sahwa</span></a>
  <div class="hmid">
    <div class="pill"><div class="dot" id="cdot"></div><span id="ctext">Waiting for device</span></div>
    <div class="mode-toggle">
      <button class="mtb aw" id="btw" onclick="switchMode(0)">&#9785; WRIST</button>
      <button class="mtb"    id="bta" onclick="switchMode(1)">&#128694; ANKLE</button>
    </div>
    <div class="ipbadge" id="ipb">ESP: <span id="ipv"></span></div>
  </div>
  <div class="hright">
    <button class="hbtn hbtn-p" onclick="window.location.href='/api/report'">&#128196; PDF Report</button>
    <button class="hbtn hbtn-o" onclick="window.location.href='/'">&#8592; Setup</button>
  </div>
</header>

<div class="main">
  <div class="apanel" id="apanel">
    <div class="albl">CURRENT ACTIVITY</div>
    <div id="aword" style="color:var(--acc)">WAITING...</div>
    <div class="cbar-wrap"><div class="cbar-fill" id="cbf"></div></div>
    <div class="atime" id="atime">--:--:--</div>
  </div>

  <div class="charts">
    <div class="cpanel"><div class="ctitle">ACCELEROMETER (g)</div><div class="ccanvas"><canvas id="acc"></canvas></div></div>
    <div class="cpanel"><div class="ctitle">GYROSCOPE (deg/s)</div><div class="ccanvas"><canvas id="gyr"></canvas></div></div>
  </div>

  <div class="logpanel">
    <div class="lphead">
      <div class="ctitle">EPISODE LOG</div>
      <button class="lpbtn" onclick="clearLog()">&#128465; Clear</button>
    </div>
    <div class="logscroll">
      <table>
        <thead><tr><th>TIME</th><th>EVENT</th><th>DURATION</th><th>SEVERITY</th></tr></thead>
        <tbody id="tbody"><tr><td colspan="4" style="text-align:center;padding:16px;color:var(--t3)">No episodes recorded</td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="sidebar">
    <div class="spanel">
      <div class="shead">PATIENT</div>
      <div class="patcard"><div class="avatar">&#128100;</div><div><div class="pname" id="pname">Patient</div><div class="psub" id="psub">WRIST &bull; EPILEPSY / FALL MODE</div></div></div>
      <div class="emlrow" id="emlrow">&#128231; <span id="emlv"></span></div>
      <div class="buflbl"><span>DATA BUFFER</span><span id="bufpct">0%</span></div>
      <div class="buftrack"><div class="buffill" id="buffill"></div></div>
      <div class="samplbl">SAMPLES: <span id="scount" style="color:var(--acc)">0</span></div>
    </div>
    <div class="spanel">
      <div class="shead">STATISTICS</div>
      <div class="sgrid">
        <div class="sbox"><div class="sval" id="stot">0</div><div class="slbl">EPISODES</div></div>
        <div class="sbox"><div class="sval" id="slong">0s</div><div class="slbl">LONGEST</div></div>
        <div class="sbox sfull"><div class="sval" id="sfreq" style="font-size:14px">N/A</div><div class="slbl">MOST FREQUENT</div></div>
      </div>
    </div>
    <div class="spanel">
      <div class="shead">CONTROLS</div>
      <button class="cbtn cbtn-p" onclick="window.location.href='/api/report'">&#128196; Generate PDF Report</button>
      <button class="cbtn cbtn-o" onclick="clearLog()">&#128465; Clear Episode Log</button>
      <div class="devnotes"><p>Short press button = switch mode<br>Long press 2s = test alert</p></div>
    </div>
  </div>
</div>

<div id="notif"></div>

<script>
const params = new URLSearchParams(location.search);
const pname  = params.get('name')  || 'Patient';
const pemail = params.get('email') || '';
const initMode = parseInt(params.get('mode') || '0');

document.getElementById('pname').textContent = pname;
if (pemail) {
  document.getElementById('emlrow').style.display = 'block';
  document.getElementById('emlv').textContent = pemail;
}

// ── CHARTS ────────────────────────────────────────────────────────────────
const N = 200;
function makeChart(id, labels, colors) {
  return new Chart(document.getElementById(id).getContext('2d'), {
    type: 'line',
    data: {
      labels: Array(N).fill(''),
      datasets: labels.map((l,i) => ({
        label: l, data: Array(N).fill(null),
        borderColor: colors[i], borderWidth: 1.5,
        pointRadius: 0, tension: 0.3, fill: false
      }))
    },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { ticks: { color:'#405d7a', font:{family:'JetBrains Mono',size:9} },
             grid: { color:'rgba(22,38,61,0.6)' } }
      },
      plugins: { legend: { labels: { color:'#405d7a', font:{family:'JetBrains Mono',size:9}, boxWidth:8 } } }
    }
  });
}
const accChart = makeChart('acc', ['X','Y','Z'], ['#ef4444','#22c55e','#0ea5e9']);
const gyrChart = makeChart('gyr', ['X','Y','Z'], ['#f97316','#a855f7','#06d6a0']);

function pushChart(chart, vals) {
  chart.data.datasets.forEach((ds,i) => { ds.data.push(vals[i]); if(ds.data.length>N) ds.data.shift(); });
  chart.update('none');
}

let sampleCount = 0, bufCount = 0;

// ── SSE ───────────────────────────────────────────────────────────────────
function connectSSE() {
  const es = new EventSource('/events');

  es.addEventListener('imu', e => {
    const d = JSON.parse(e.data);
    pushChart(accChart, [d.ax, d.ay, d.az]);
    pushChart(gyrChart, [d.gx, d.gy, d.gz]);
    sampleCount++;
    bufCount = Math.min(bufCount+1, 256);
    document.getElementById('scount').textContent = sampleCount;
    const pct = Math.round(bufCount/256*100);
    document.getElementById('buffill').style.width = pct+'%';
    document.getElementById('bufpct').textContent  = pct+'%';
  });

  es.addEventListener('activity', e => {
    const d = JSON.parse(e.data);
    const aw = document.getElementById('aword');
    const ap = document.getElementById('apanel');
    aw.textContent = d.activity;
    aw.style.color = d.color;
    aw.style.textShadow = d.critical ? '0 0 40px '+d.color : 'none';
    aw.classList.toggle('crit', d.critical);
    ap.classList.toggle('crit', d.critical);
    const cf = document.getElementById('cbf');
    cf.style.width = '92%'; cf.style.background = d.color;
    document.getElementById('atime').textContent = new Date().toLocaleTimeString();
    bufCount = 0;
    setTimeout(() => { cf.style.width = '0%'; }, 2600);
  });

  es.addEventListener('status', e => {
    const d = JSON.parse(e.data);
    const dot = document.getElementById('cdot');
    dot.className = d.connected ? 'dot on' : 'dot';
    document.getElementById('ctext').textContent = d.connected ? 'ESP32 Connected' : 'Waiting for device';
    if (d.ip) {
      document.getElementById('ipb').style.display = 'inline-block';
      document.getElementById('ipv').textContent = d.ip;
    }
    updateModeUI(d.mode);
  });

  es.addEventListener('alert', e => {
    const d = JSON.parse(e.data);
    const b = document.getElementById('banner');
    document.getElementById('btext').textContent = d.type + ' at ' + d.time;
    b.classList.add('on');
    setTimeout(() => b.classList.remove('on'), 12000);
  });

  es.addEventListener('log', e => {
    const d = JSON.parse(e.data);
    renderLog(d.log);
  });

  es.addEventListener('stats', e => {
    const d = JSON.parse(e.data);
    document.getElementById('stot').textContent  = d.total;
    document.getElementById('slong').textContent = d.longest + 's';
    document.getElementById('sfreq').textContent = d.freq;
  });

  let notifTimer = null;
  es.addEventListener('notif', e => {
    const d = JSON.parse(e.data);
    const el = document.getElementById('notif');
    el.textContent = d.msg;
    el.className = 'on n' + (d.t||'i')[0];
    clearTimeout(notifTimer);
    notifTimer = setTimeout(() => el.classList.remove('on'), 3500);
  });

  es.onerror = () => {
    es.close();
    setTimeout(connectSSE, 3000);
  };
}
connectSSE();

function updateModeUI(m) {
  document.getElementById('btw').className = 'mtb' + (m===0 ? ' aw' : '');
  document.getElementById('bta').className = 'mtb' + (m===1 ? ' aa' : '');
  document.getElementById('psub').textContent = m===0
    ? 'WRIST \u2022 EPILEPSY / FALL MODE'
    : 'ANKLE \u2022 PARKINSON\u2019S / FOG MODE';
}

function switchMode(m) {
  fetch('/api/set_mode', { method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mode:m}) });
  updateModeUI(m);
}

function renderLog(log) {
  const tb = document.getElementById('tbody');
  if (!log || !log.length) {
    tb.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:16px;color:var(--t3)">No episodes recorded</td></tr>';
    return;
  }
  tb.innerHTML = '';
  [...log].reverse().forEach(ep => {
    const cls = ep.severity === 'CRITICAL' ? 'bc' : 'bw';
    const tr  = document.createElement('tr');
    tr.innerHTML = '<td>'+ep.time+'</td><td>'+ep.type+'</td><td>'+ep.dur+'s</td>'
      + '<td><span class="badge '+cls+'">'+ep.severity+'</span></td>';
    tb.appendChild(tr);
  });
}

function clearLog() {
  if (!confirm('Clear all episode records?')) return;
  fetch('/api/clear_log', {method:'POST'});
}

updateModeUI(initMode);
</script></body></html>"""


@app.route("/")
def index():
    return render_template_string(LOGIN_HTML)

@app.route("/dashboard")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 56)
    print("  sahwa v4.0 — Full Stack Server")
    print("  Abu Dhabi University URIC 2026")
    print("=" * 56)
    load_episodes()
    models = load_models()
    TCPServer(models).start()
    Inference(models).start()
    url = "http://127.0.0.1:{}".format(WEB_PORT)
    print("[Web] http://127.0.0.1:{}".format(WEB_PORT))
    print("[TCP] Listening on :{}".format(TCP_PORT))
    print("[Info] Hotspot must be ON (ysfthecreator)")
    print("=" * 56)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=WEB_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
