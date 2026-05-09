#include <DHT.h>

#define RELAY_PIN 27
#define DHT1_PIN 26
#define DHT2_PIN 33
#define DHT_TYPE DHT22

#define TEMP_ON  19.0
#define TEMP_OFF 20.0

DHT dht1(DHT1_PIN, DHT_TYPE);
DHT dht2(DHT2_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW); // heater off by default
  dht1.begin();
  dht2.begin();
  Serial.println("Incubator starting...");
}

void loop() {
  delay(2000); // DHT22 needs 2s between readings

  float temp1 = dht1.readTemperature();
  float temp2 = dht2.readTemperature();
  float hum1  = dht1.readHumidity();
  float hum2  = dht2.readHumidity();

  // check for failed readings
  if (isnan(temp1) || isnan(temp2)) {
    Serial.println("DHT read failed! Check wiring.");
    return;
  }

  float avgTemp = (temp1 + temp2) / 2.0;

  // bang-bang thermostat
  if (avgTemp < TEMP_ON) {
    digitalWrite(RELAY_PIN, HIGH); // heater ON
  } else if (avgTemp > TEMP_OFF) {
    digitalWrite(RELAY_PIN, LOW);  // heater OFF
  }

  Serial.println("======================");
  Serial.print("DHT1 Temp: "); Serial.print(temp1); Serial.println(" C");
  Serial.print("DHT2 Temp: "); Serial.print(temp2); Serial.println(" C");
  Serial.print("Avg Temp:  "); Serial.print(avgTemp); Serial.println(" C");
  Serial.print("DHT1 Hum:  "); Serial.print(hum1); Serial.println(" %");
  Serial.print("DHT2 Hum:  "); Serial.print(hum2); Serial.println(" %");
  Serial.print("Heater:    "); Serial.println(digitalRead(RELAY_PIN) ? "ON" : "OFF");
}