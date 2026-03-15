/*
 * =================================================================
 *  sahwa Project: ESP32-C3 Real-Time IMU Data Streamer v3
 *
 *  Connects to a WiFi hotspot and streams raw MPU6500 data
 *  to a TCP server for live inference. This version includes
 *  detailed status feedback, mode switching, and test alerts.
 * =================================================================
 */

#include <Wire.h>
#include <U8g2lib.h>
#include <FastIMU.h>
#include <WiFi.h>

// ── Pin Definitions ─────────────────────────────────────────
#define SDA_PIN    5
#define SCL_PIN    6
#define RED_PIN    4
#define GREEN_PIN  7
#define BLUE_PIN   8
#define BUZZER_PIN 3
#define BUTTON_PIN 2 

// ── WiFi & TCP Config ────────────────────────────────────────
const char* WIFI_SSID   = "ysfthecreator";
const char* WIFI_PASS   = "ysfthecreator@123";
const char* TCP_HOST_IP = "192.168.137.1";
const int   TCP_PORT    = 8888;

WiFiClient client;

// ── IMU (MPU6500) Config ─────────────────────────────────────
#define IMU_ADDRESS 0x68
MPU6500 IMU;
AccelData accel;
GyroData gyro;
calData calib;

// ── OLED (SSD1306 72x40) ─────────────────────────────────────
U8G2_SSD1306_72X40_ER_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, SCL_PIN, SDA_PIN);

// ── System State & Modes ─────────────────────────────────────
enum SystemState { STATE_WIFI_CONNECTING, STATE_SERVER_WAITING, STATE_STREAMING };
SystemState currentState = STATE_WIFI_CONNECTING;

enum DeviceMode { MODE_WRIST = 0, MODE_ANKLE = 1 };
DeviceMode currentMode = MODE_WRIST;
const char* MODE_NAMES[] = {"Wrist", "Ankle"};
bool modeChanged = true; // Flag to send mode on connect

// ── Button Handling ──────────────────────────────────────────
volatile bool buttonPressed = false;
unsigned long buttonPressTime = 0;
unsigned long lastButtonCheck = 0;
const int DEBOUNCE_MS = 50;

// ── Timing & Counters ────────────────────────────────────────
#define SAMPLE_INTERVAL_MS 20 // 50 Hz
unsigned long lastSampleTime = 0;
unsigned long sampleCounter = 0;

// Function Prototypes
void setRGB(int r, int g, int b);
void updateDisplay();
void beep(int hz, int ms);
void handleButton();
void connectWiFi();
void connectToServer();
void streamIMUData();


void setup() {
  Serial.begin(115200);

  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(BUZZER_PIN, LOW);

  u8g2.begin();
  u8g2.setFont(u8g2_font_5x8_tr);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  int err = IMU.init(calib, IMU_ADDRESS);
  if (err != 0) {
    while (true) { setRGB(255, 0, 0); delay(100); setRGB(0, 0, 0); delay(100); }
  }

  connectWiFi();
}

void loop() {
  handleButton();
  
  switch (currentState) {
    case STATE_WIFI_CONNECTING:
      connectWiFi();
      break;
    case STATE_SERVER_WAITING:
      if (WiFi.status() != WL_CONNECTED) {
        currentState = STATE_WIFI_CONNECTING;
        beep(400, 50); delay(60); beep(400, 50); delay(60); beep(400, 50);
      } else {
        connectToServer();
        delay(1000);
      }
      break;
    case STATE_STREAMING:
      if (!client.connected()) {
        currentState = STATE_SERVER_WAITING;
        beep(400, 50); delay(60); beep(400, 50); delay(60); beep(400, 50);
        break;
      }
      if (millis() - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = millis();
        streamIMUData();
      }
      break;
  }
  updateDisplay();
}

void handleButton() {
    if (millis() - lastButtonCheck < DEBOUNCE_MS) {
        return;
    }
    lastButtonCheck = millis();
    bool buttonState = digitalRead(BUTTON_PIN);

    if (buttonState == LOW) { // Button is pressed
        if (buttonPressTime == 0) { // First press
            buttonPressTime = millis();
        }
    } else { // Button is released
        if (buttonPressTime > 0) {
            unsigned long pressDuration = millis() - buttonPressTime;
            if (pressDuration < 1000) { // Short press
                currentMode = (currentMode == MODE_WRIST) ? MODE_ANKLE : MODE_WRIST;
                modeChanged = true; // Flag to send update
                beep(1800, 50);
            }
        }
        buttonPressTime = 0; // Reset
    }

    // Long press check
    if (buttonPressTime > 0 && (millis() - buttonPressTime > 2000)) {
        if (currentState == STATE_STREAMING) {
            client.println("TEST_ALERT");
        }
        beep(2000, 150); delay(50); beep(2000, 150);
        buttonPressTime = 0; // Reset after action to prevent repeats
    }
}

void connectWiFi() {
  currentState = STATE_WIFI_CONNECTING;
  updateDisplay(); // Shows "WiFi: Connecting"
  
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500); retries++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    ESP.restart();
  } else {
    currentState = STATE_SERVER_WAITING;
    beep(1200, 80);
  }
}

void connectToServer() {
  updateDisplay(); // Shows "Server: Waiting"
  
  if (client.connect(TCP_HOST_IP, TCP_PORT)) {
    currentState = STATE_STREAMING;
    sampleCounter = 0;
    modeChanged = true; // Ensure mode is sent on new connection
    beep(1500, 60); delay(70); beep(1500, 60);
  }
}

void streamIMUData() {
  // If mode has changed, send the update first
  if (modeChanged && client.connected()) {
      client.print("MODE:");
      client.println(currentMode);
      modeChanged = false;
  }
    
  IMU.update();
  IMU.getAccel(&accel);
  IMU.getGyro(&gyro);

  String dataLine = String(millis()) + "," + String(accel.accelX, 4) + "," + String(accel.accelY, 4) + "," + String(accel.accelZ, 4) + "," +
                    String(gyro.gyroX, 4) + "," + String(gyro.gyroY, 4) + "," + String(gyro.gyroZ, 4);
  
  client.println(dataLine);
  sampleCounter++;

  if (sampleCounter % 50 == 0) {
    setRGB(255, 255, 255); delay(100);
  }
}

void updateDisplay() {
    static unsigned long lastDisplayTime = 0;
    if (millis() - lastDisplayTime < 250) return; // Limit updates to 4Hz
    lastDisplayTime = millis();

    u8g2.clearBuffer();
    u8g2.drawStr(0, 7, "sahwa");

    char modeLine[20];
    snprintf(modeLine, sizeof(modeLine), "Mode: %s", MODE_NAMES[currentMode]);
    u8g2.drawStr(0, 17, modeLine);

    char statusLine[20];
    switch (currentState) {
        case STATE_WIFI_CONNECTING:
            strcpy(statusLine, "WiFi: Connecting");
            setRGB(255, 0, 0); // Red
            break;
        case STATE_SERVER_WAITING:
            strcpy(statusLine, "Server: Waiting");
            if (currentMode == MODE_WRIST) setRGB(0, 255, 255); // Cyan
            else setRGB(128, 0, 128); // Purple
            break;
        case STATE_STREAMING:
            strcpy(statusLine, "Status: Streaming");
            // Set base color, allow heartbeat to override
            if (currentMode == MODE_WRIST) setRGB(0, 255, 255); // Cyan
            else setRGB(128, 0, 128); // Purple
            break;
    }
    u8g2.drawStr(0, 27, statusLine);

    char countStr[20];
    snprintf(countStr, sizeof(countStr), "S: %lu", sampleCounter);
    u8g2.drawStr(0, 37, countStr);

    u8g2.sendBuffer();
}

void setRGB(int r, int g, int b) {
  analogWrite(RED_PIN, r);
  analogWrite(GREEN_PIN, g);
  analogWrite(BLUE_PIN, b);
}

void beep(int hz, int ms) {
  tone(BUZZER_PIN, hz, ms);
}
