#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <U8g2lib.h>
#include <WiFi.h>

// ── Pin Definitions ─────────────────────────────────────────
#define PIN_R 4
#define PIN_G 7
#define PIN_B 8
#define PIN_BUZZ 3
#define PIN_BTN  2

// ── WiFi Configuration ──────────────────────────────────────
const char* SSID = "ysfthecreator";
const char* PASS = "ysfthecreator@123";

// ── Objects ──────────────────────────────────────────────────
Adafruit_MPU6050 mpu;
U8G2_SSD1306_72X40_ER_F_HW_I2C display(U8G2_R0, U8X8_PIN_NONE, 6, 5); // SCL=6, SDA=5

// ── State Variables ─────────────────────────────────────────
int  currentLabel = 0;
bool recording    = false;
unsigned long lastSample = 0;
const int SAMPLE_HZ = 50;
const int INTERVAL  = 1000 / SAMPLE_HZ;

// ── Activity Labels and Colors ──────────────────────────────
const uint8_t COLORS[9][3] = {
  {0,255,0},   // 0 Stand    - Green
  {0,255,255}, // 1 Walk     - Cyan
  {0,0,255},   // 2 FastWalk - Blue
  {255,128,0}, // 3 Sit      - Orange
  {255,255,0}, // 4 SitStand - Yellow
  {128,0,255}, // 5 Stairs   - Purple
  {255,64,0},  // 6 Fall     - Red-Orange
  {255,0,128}, // 7 FoG      - Pink
  {255,0,0}    // 8 Seizure  - Red
};

const char* NAMES[9] = {
  "Stand","Walk","FastWalk","Sit",
  "SitStand","Stairs","Fall","FoG","Seizure"
};

void setLED(uint8_t r, uint8_t g, uint8_t b) {
  analogWrite(PIN_R, r);
  analogWrite(PIN_G, g);
  analogWrite(PIN_B, b);
}

void updateDisplay() {
  display.clearBuffer();
  display.setFont(u8g2_font_6x10_tr);
  display.drawStr(0, 10, recording ? "* REC *" : "READY");
  display.drawStr(0, 25, NAMES[currentLabel]);
  display.sendBuffer();
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(PIN_R,    OUTPUT);
  pinMode(PIN_G,    OUTPUT);
  pinMode(PIN_B,    OUTPUT);
  pinMode(PIN_BUZZ, OUTPUT);
  pinMode(PIN_BTN,  INPUT_PULLUP);

  // Initialize OLED (U8G2)
  display.begin();
  display.clearBuffer();
  display.setFont(u8g2_font_6x10_tr);
  display.drawStr(0, 10, "Booting...");
  display.sendBuffer();

  // Initialize IMU
  Wire.begin(5, 6); // SDA=5, SCL=6
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (1) delay(100);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_1000_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // WiFi Connection with 10s Timeout
  WiFi.begin(SSID, PASS);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 20) {
    delay(500);
    tries++;
  }
  
  if (WiFi.status() == WL_CONNECTED)
    Serial.println("WiFi OK");
  else
    Serial.println("WiFi skipped - USB only mode");

  setLED(0, 255, 0); // Default to Green
  updateDisplay();
  Serial.println("READY");
}

void loop() {
  // Handle Serial Commands (from Python script)
  if (Serial.available()) {
    char c = Serial.read();

    if (c >= '0' && c <= '8') {
      currentLabel = c - '0';
      setLED(COLORS[currentLabel][0], COLORS[currentLabel][1], COLORS[currentLabel][2]);
      updateDisplay();
      Serial.print("LABEL:");
      Serial.println(currentLabel);
    }
    else if (c == 'S' || c == 's') {
      recording = !recording;
      if (recording) {
        Serial.println("START");
        tone(PIN_BUZZ, 1000, 100);
      } else {
        Serial.println("STOP");
        tone(PIN_BUZZ, 500, 200);
      }
      updateDisplay();
    }
  }

  // IMU Sampling at 50Hz
  if (recording && millis() - lastSample >= INTERVAL) {
    lastSample = millis();
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // CSV Format: timestamp,ax,ay,az,gx,gy,gz,label
    Serial.print(millis());     Serial.print(",");
    Serial.print(a.acceleration.x, 4); Serial.print(",");
    Serial.print(a.acceleration.y, 4); Serial.print(",");
    Serial.print(a.acceleration.z, 4); Serial.print(",");
    Serial.print(g.gyro.x, 4);  Serial.print(",");
    Serial.print(g.gyro.y, 4);  Serial.print(",");
    Serial.print(g.gyro.z, 4);  Serial.print(",");
    Serial.println(currentLabel);
  }
}
