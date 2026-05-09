/*
 * ============================================================
 *  URIC RESEARCH PROJECT: ESP32-C3 Wearable Data Logger
 *  USB SERIAL ONLY MODE - No WiFi Needed
 *  Labels: 0=Stand 1=Walk 2=FastWalk 3=Sit 4=SitStand
 *          5=Stairs 6=Fall 7=FoG 8=Seizure
 * ============================================================
 */

#include <Wire.h>
#include <U8g2lib.h>
#include <FastIMU.h>

// ── Pin Definitions ─────────────────────────────────────────
#define SDA_PIN    5
#define SCL_PIN    6
#define RED_PIN    4
#define GREEN_PIN  7
#define BLUE_PIN   8
#define BUZZER_PIN 3
#define BUTTON_PIN 2

#define MPU6500_ADDRESS 0x68

// ── OLED ─────────────────────────────────────────────────────
U8G2_SSD1306_72X40_ER_F_HW_I2C u8g2(U8G2_R0, U8X8_PIN_NONE, SCL_PIN, SDA_PIN);

// ── IMU ──────────────────────────────────────────────────────
MPU6500 IMU;
AccelData accel;
GyroData gyro;
calData calib;

// ── Labels ───────────────────────────────────────────────────
#define NUM_LABELS 9
const char* labelNames[] = {
  "Stand",     // 0
  "Walk",      // 1
  "FastWalk",  // 2
  "Sit",       // 3
  "SitStand",  // 4
  "Stairs",    // 5
  "Fall",      // 6
  "FoG",       // 7
  "Seizure"    // 8
};

// LED colors per label [R, G, B]
const int labelColors[NUM_LABELS][3] = {
  {0,   255, 0  },  // 0 Stand     - Green
  {0,   200, 255},  // 1 Walk      - Cyan
  {0,   100, 255},  // 2 FastWalk  - Blue
  {255, 165, 0  },  // 3 Sit       - Orange
  {255, 200, 0  },  // 4 SitStand  - Yellow
  {128, 0,   255},  // 5 Stairs    - Purple
  {255, 50,  0  },  // 6 Fall      - Red-Orange
  {255, 0,   150},  // 7 FoG       - Pink
  {255, 0,   0  },  // 8 Seizure   - Red
};

// ── Logging Config ───────────────────────────────────────────
#define SAMPLE_RATE_MS  20   // 50 Hz

uint8_t currentLabel = 0;
bool logging         = false;
unsigned long lastSampleTime  = 0;
unsigned long sampleCount     = 0;
unsigned long sessionStart    = 0;
unsigned long lastButtonTime  = 0;
unsigned long lastDisplayUpdate = 0;
bool lastButtonState = HIGH;
unsigned long buttonPressStart = 0;

// ── Send line via USB Serial ────────────────────────────────
void sendLine(const String& s) {
  Serial.println(s);
}

// ── Setup ────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(RED_PIN,    OUTPUT);
  pinMode(GREEN_PIN,  OUTPUT);
  pinMode(BLUE_PIN,   OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(BUZZER_PIN, LOW);

  setRGB(0, 0, 255);

  u8g2.begin();
  showMessage("URIC", "Wearable", "USB ONLY");
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  showMessage("Init", "IMU...", "");
  int err = IMU.init(calib, MPU6500_ADDRESS);
  if (err != 0) {
    showMessage("IMU", "FAIL", String(err).c_str());
    while (1) { setRGB(255,0,0); delay(300); setRGB(0,0,0); delay(300); }
  }

  setRGB(255, 165, 0); 
  setLabelColor();
  updateDisplay();
  Serial.println("READY");
}

// ── Main Loop ────────────────────────────────────────────────
void loop() {
  // Handle commands from USB Serial
  if (Serial.available()) {
    handleCommand(Serial.read());
  }

  // Sample IMU
  if (logging && (millis() - lastSampleTime >= SAMPLE_RATE_MS)) {
    lastSampleTime = millis();
    takeSample();
  }

  handleButton();

  // Update display every 500ms
  if (millis() - lastDisplayUpdate > 500) {
    lastDisplayUpdate = millis();
    updateDisplay();
  }
}

// ── Take IMU sample and send ─────────────────────────────────
void takeSample() {
  IMU.update();
  IMU.getAccel(&accel);
  IMU.getGyro(&gyro);

  float ax = accel.accelX, ay = accel.accelY, az = accel.accelZ;
  float gx = gyro.gyroX,   gy = gyro.gyroY,   gz = gyro.gyroZ;

  String line = String(millis()) + "," +
                String(ax, 4) + "," + String(ay, 4) + "," + String(az, 4) + "," +
                String(gx, 4) + "," + String(gy, 4) + "," + String(gz, 4) + "," +
                String(currentLabel) + "," + String(labelNames[currentLabel]);
  sendLine(line);
  sampleCount++;
}

// ── Handle command from laptop ───────────────────────────────
void handleCommand(char cmd) {
  // Labels 0-8 via number keys
  if (cmd >= '0' && cmd <= '8') {
    currentLabel = cmd - '0';
    setLabelColor();
    sendLine("# Label set: " + String(labelNames[currentLabel]));
  }
  // S = start/stop
  else if (cmd == 'S' || cmd == 's') {
    toggleLogging();
  }
  // C = calibrate
  else if (cmd == 'C' || cmd == 'c') {
    calibrateIMU();
  }
  // R = reset
  else if (cmd == 'R' || cmd == 'r') {
    sampleCount = 0;
    sendLine("# Session reset");
  }
}

// ── Toggle recording ─────────────────────────────────────────
void toggleLogging() {
  logging = !logging;
  if (logging) {
    sessionStart = millis();
    sampleCount  = 0;
    sendLine("# === SESSION START ===");
    beep(1000, 100);
  } else {
    sendLine("# === SESSION END ===");
    beep(500, 300);
  }
}

// ── Calibrate IMU ────────────────────────────────────────────
void calibrateIMU() {
  sendLine("# Calibrating... keep still for 10s");
  showMessage("CAL", "Keep", "Still!");
  setRGB(255, 255, 0);
  IMU.calibrateAccelGyro(&calib);
  IMU.init(calib, MPU6500_ADDRESS);
  sendLine("# Calibration done!");
  beep(1500, 200);
  setLabelColor();
}

// ── Button handler ───────────────────────────────────────────
void handleButton() {
  bool state = digitalRead(BUTTON_PIN);
  unsigned long now = millis();

  if (state == LOW && lastButtonState == HIGH && (now - lastButtonTime > 200)) {
    lastButtonTime   = now;
    buttonPressStart = now;
  }

  // Short press → cycle label
  if (state == HIGH && lastButtonState == LOW) {
    if ((millis() - buttonPressStart) < 1500) {
      currentLabel = (currentLabel + 1) % NUM_LABELS;
      setLabelColor();
      beep(800, 50);
    }
  }

  // Long press → toggle recording
  if (state == LOW && lastButtonState == LOW) {
    if ((millis() - buttonPressStart) > 1500 && (millis() - lastButtonTime) > 2000) {
      lastButtonTime = millis();
      toggleLogging();
    }
  }

  lastButtonState = state;
}

// ── Set LED color based on current label ─────────────────────
void setLabelColor() {
  if (currentLabel < NUM_LABELS) {
    setRGB(labelColors[currentLabel][0],
           labelColors[currentLabel][1],
           labelColors[currentLabel][2]);
  }
}

// ── Update OLED display ──────────────────────────────────────
void updateDisplay() {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_micro_tr);

  // Line 1: status
  if (logging) {
    u8g2.drawStr(0, 8, "* REC *");
  } else {
    u8g2.drawStr(0, 8, "READY (USB)");
  }

  // Line 2: label number + name
  u8g2.setFont(u8g2_font_6x10_tr);
  u8g2.setCursor(0, 22);
  u8g2.print(currentLabel);
  u8g2.print(":");
  u8g2.print(labelNames[currentLabel]);

  // Line 3: sample count
  u8g2.setFont(u8g2_font_micro_tr);
  u8g2.setCursor(0, 36);
  u8g2.print(sampleCount);
  u8g2.print(" smp");

  u8g2.sendBuffer();
}

// ── Utilities ────────────────────────────────────────────────
void showMessage(const char* l1, const char* l2, const char* l3) {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tr);
  u8g2.drawStr(0, 12, l1);
  u8g2.drawStr(0, 24, l2);
  u8g2.setFont(u8g2_font_micro_tr);
  u8g2.drawStr(0, 36, l3);
  u8g2.sendBuffer();
}

void beep(int hz, int ms) { tone(BUZZER_PIN, hz, ms); }

void setRGB(int r, int g, int b) {
  analogWrite(RED_PIN,   r);
  analogWrite(GREEN_PIN, g);
  analogWrite(BLUE_PIN,  b);
}
