/*
 * sahwa ESP32-C3 Firmware — Cloud / WebSocket version
 *
 * Changes from local TCP version:
 *   - Connects to your own WiFi (not laptop hotspot)
 *   - Streams IMU data to Railway server via WebSocket
 *   - No hotspot needed, works from any WiFi anywhere
 *
 * Library: ArduinoWebsockets by Gil Maimon
 *   Install: Arduino IDE → Library Manager → search "ArduinoWebsockets"
 *
 * Other libraries (same as before):
 *   FastIMU, U8g2, Wire (built-in)
 */

#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <Wire.h>
#include "FastIMU.h"
#include <U8g2lib.h>

using namespace websockets;

// ─── CONFIGURE THESE ─────────────────────────────────────────────────────
const char* WIFI_SSID  = "ysfthecreator";
const char* WIFI_PASS  = "ysfthecreator@123";
const char* SERVER_URL = "wss://sahwa-monitor.up.railway.app/esp32";
//  ^ Replace YOUR-APP with your actual Railway app name
//    e.g. "wss://sahwa-production.up.railway.app/esp32"
// ─────────────────────────────────────────────────────────────────────────

// Pins (same as before)
#define SDA_PIN   5
#define SCL_PIN   6
#define BTN_PIN   2
#define LED_R     4
#define LED_G     7
#define LED_B     8
#define BUZZER    3

// Objects
WebsocketsClient wsClient;
MPU6500 imu;
#define IMU_ADDRESS 0x68
U8G2_SSD1306_72X40_ER_F_HW_I2C display(U8G2_R0, U8X8_PIN_NONE);

// State
calData    cal = { 0 };
AccelData  accelData;
GyroData   gyroData;

int  deviceMode    = 0;   // 0 = Wrist, 1 = Ankle
bool wsConnected   = false;
int  sampleCount   = 0;

unsigned long lastSample   = 0;
unsigned long lastReconnect = 0;
unsigned long btnPressTime = 0;
bool          btnHeld      = false;

const unsigned long SAMPLE_INTERVAL = 20;   // 50 Hz
const unsigned long RECONNECT_MS    = 5000;

// ─── LED HELPERS ─────────────────────────────────────────────────────────
void setRGB(int r, int g, int b) {
  // Common anode — LOW = on
  analogWrite(LED_R, 255 - r);
  analogWrite(LED_G, 255 - g);
  analogWrite(LED_B, 255 - b);
}
void ledOff()    { setRGB(0,   0,   0); }
void ledRed()    { setRGB(255, 0,   0); }
void ledCyan()   { setRGB(0,   200, 255); }   // Wrist mode
void ledPurple() { setRGB(180, 0,   255); }   // Ankle mode
void ledWhite()  { setRGB(255, 255, 255); }

void beep(int n) {
  for (int i = 0; i < n; i++) {
    tone(BUZZER, 1000, 120);
    delay(200);
  }
}

// ─── DISPLAY ─────────────────────────────────────────────────────────────
void updateDisplay(const char* line1, const char* line2, const char* line3 = "") {
  display.clearBuffer();
  display.setFont(u8g2_font_6x10_tf);
  display.drawStr(0, 10, line1);
  display.drawStr(0, 22, line2);
  if (strlen(line3) > 0) display.drawStr(0, 34, line3);
  display.sendBuffer();
}

// ─── WEBSOCKET CALLBACKS ─────────────────────────────────────────────────
void onMessage(WebsocketsMessage msg) {
  String data = msg.data();
  if (data == "ALERT") {
    // Server confirmed critical event — buzz the device
    tone(BUZZER, 2000, 150); delay(200);
    tone(BUZZER, 2000, 150); delay(200);
    tone(BUZZER, 2000, 150);
    ledRed();
    delay(500);
    setModeColor();
  } else if (data.startsWith("SET_MODE:")) {
    deviceMode = data.substring(9).toInt();
    setModeColor();
  }
}

void onEvent(WebsocketsEvent event, String data) {
  if (event == WebsocketsEvent::ConnectionOpened) {
    wsConnected = true;
    Serial.println("[WS] Connected to server");
    beep(2);
    setModeColor();
    updateDisplay("sahwa", wsConnected ? "Connected" : "...", deviceMode == 0 ? "Wrist Mode" : "Ankle Mode");
    // Tell server initial mode
    wsClient.send("MODE:" + String(deviceMode));
  } else if (event == WebsocketsEvent::ConnectionClosed) {
    wsConnected = false;
    Serial.println("[WS] Disconnected");
    beep(3);
    ledRed();
    updateDisplay("sahwa", "Disconnected", "Reconnecting...");
  } else if (event == WebsocketsEvent::GotPing) {
    wsClient.pong();
  }
}

void setModeColor() {
  if (wsConnected) {
    deviceMode == 0 ? ledCyan() : ledPurple();
  }
}

// ─── WIFI + WS CONNECTION ─────────────────────────────────────────────────
void connectWiFi() {
  Serial.print("[WiFi] Connecting to "); Serial.println(WIFI_SSID);
  ledRed();
  updateDisplay("sahwa", "WiFi...", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500); Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected: " + WiFi.localIP().toString());
    beep(1);
    updateDisplay("WiFi OK", WiFi.localIP().toString().c_str(), "");
    delay(1000);
  } else {
    Serial.println("\n[WiFi] Failed! Check SSID/password.");
    updateDisplay("WiFi FAIL", "Check creds", "Rebooting...");
    delay(3000);
    ESP.restart();
  }
}

void connectWS() {
  Serial.println("[WS] Connecting to " + String(SERVER_URL));
  updateDisplay("sahwa", "Connecting WS", "...");
  wsClient.onMessage(onMessage);
  wsClient.onEvent(onEvent);
  wsClient.connect(SERVER_URL);
}

// ─── SETUP ────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  pinMode(BTN_PIN, INPUT_PULLUP);
  pinMode(LED_R, OUTPUT);
  pinMode(LED_G, OUTPUT);
  pinMode(LED_B, OUTPUT);
  ledRed();

  Wire.begin(SDA_PIN, SCL_PIN);

  // OLED
  display.begin();
  display.setFont(u8g2_font_6x10_tf);
  updateDisplay("sahwa", "Starting...", "v4.0 Cloud");
  delay(500);

  // IMU
  int err = imu.init(cal, IMU_ADDRESS);
  if (err != 0) {
    Serial.println("[IMU] Init failed: " + String(err));
    updateDisplay("IMU ERROR", String(err).c_str(), "Check wiring");
    while (true) { delay(1000); }
  }
  Serial.println("[IMU] Ready");

  connectWiFi();
  connectWS();
}

// ─── LOOP ─────────────────────────────────────────────────────────────────
void loop() {
  wsClient.poll();

  // Reconnect if dropped
  if (!wsConnected && millis() - lastReconnect > RECONNECT_MS) {
    lastReconnect = millis();
    Serial.println("[WS] Reconnecting...");
    connectWS();
  }

  // Button handling
  bool btnPressed = (digitalRead(BTN_PIN) == LOW);
  if (btnPressed && !btnHeld) {
    btnPressTime = millis();
    btnHeld = true;
  }
  if (!btnPressed && btnHeld) {
    unsigned long held = millis() - btnPressTime;
    btnHeld = false;
    if (held >= 2000) {
      // Long press → test alert
      Serial.println("[BTN] Long press — TEST_ALERT");
      if (wsConnected) wsClient.send("TEST_ALERT");
      tone(BUZZER, 1500, 300);
    } else {
      // Short press → toggle mode
      deviceMode = 1 - deviceMode;
      Serial.println("[BTN] Mode -> " + String(deviceMode == 0 ? "Wrist" : "Ankle"));
      if (wsConnected) wsClient.send("MODE:" + String(deviceMode));
      setModeColor();
      updateDisplay("sahwa", wsConnected ? "Connected" : "Disconnected",
                    deviceMode == 0 ? "Wrist Mode" : "Ankle Mode");
    }
  }

  // IMU streaming at 50Hz
  if (wsConnected && millis() - lastSample >= SAMPLE_INTERVAL) {
    lastSample = millis();
    imu.update();
    imu.getAccel(&accelData);
    imu.getGyro(&gyroData);

    // Build CSV line: ts,ax,ay,az,gx,gy,gz
    String line = String(millis()) + "," +
                  String(accelData.accelX, 4) + "," +
                  String(accelData.accelY, 4) + "," +
                  String(accelData.accelZ, 4) + "," +
                  String(gyroData.gyroX, 4)   + "," +
                  String(gyroData.gyroY, 4)   + "," +
                  String(gyroData.gyroZ, 4);

    wsClient.send(line);
    sampleCount++;

    // Heartbeat display every 100 samples
    if (sampleCount % 100 == 0) {
      ledWhite();
      delay(20);
      setModeColor();
      updateDisplay("sahwa",
                    wsConnected ? "Connected" : "Disconnected",
                    deviceMode == 0 ? "Wrist" : "Ankle");
    }
  }
}
