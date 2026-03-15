# sahwa — Cloud Deployment Guide

## Step 1: Deploy to Railway (5 minutes)

1. Go to https://github.com and create a free account
2. Create a new repo called "sahwa"
3. Upload these files to the repo:
   - sahwa_server.py
   - requirements.txt
   - Procfile
   - railway.json
   - wrist_rf_model.pkl  ← from your project folder
   - wrist_rf_scaler.pkl
   - wrist_label_mapping.json
   - ankle_rf_model.pkl
   - ankle_rf_scaler.pkl
   - ankle_label_mapping.json
   - episodes.json  (can be empty: [])

4. Go to https://railway.app → Sign in with GitHub
5. Click "New Project" → "Deploy from GitHub repo" → select "sahwa"
6. Railway auto-detects Python and deploys

7. Click "Variables" tab → add these:
   GMAIL_SENDER        = yossefgumball@gmail.com
   GMAIL_APP_PASSWORD  = (generate new app password first!)
   CAREGIVER_EMAIL     = (whoever receives alerts)

8. Click "Settings" → copy your public URL
   It looks like: https://sahwa-production.up.railway.app

Your dashboard is now live at that URL — accessible from any device!

---

## Step 2: Update ESP32 Firmware

Open sahwa_stream_cloud.ino and change:

   const char* WIFI_SSID  = "YOUR_WIFI_NAME";
   const char* WIFI_PASS  = "YOUR_WIFI_PASSWORD";
   const char* SERVER_URL = "wss://sahwa-production.up.railway.app/esp32";

Install one extra library in Arduino IDE:
   Library Manager → search "ArduinoWebsockets" by Gil Maimon → Install

Flash to ESP32-C3 (same settings as before).

---

## Step 3: Test

1. Open https://sahwa-production.up.railway.app from any phone or laptop
2. Enter patient name and caregiver email → Start Monitoring
3. Power on ESP32 — it connects to your WiFi, then to Railway
4. Dashboard shows "ESP32 Connected" immediately
5. Walk around, try the button, test an alert

---

## Important: Gmail App Password

Your old password was exposed in the uploaded file.
1. Go to myaccount.google.com/apppasswords
2. Delete the old one
3. Generate a new one named "sahwa Railway"
4. Put the new 16-char password in Railway Variables

---

## File notes

- Model files stay on Railway's disk (they get bundled with the deploy)
- episodes.json is stored on Railway's ephemeral disk (resets on redeploy)
  → For permanent storage, connect a Railway Postgres DB (future improvement)
- Free tier: 500 hours/month — enough for continuous monitoring
