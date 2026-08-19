#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <mpu6050.h>
#include <TinyGPSPlus.h>

// -------- WIFI --------
const char* ssid     = "SSID";
const char* password = "PASSWORD";

// -------- SERVER --------
const char* serverUrl = "http://localhost:8000/sensors";

// -------- MPU6050 --------
#define MPU_ADDRESS 0x68
float rawAX, rawAY, rawAZ;
float gForceAX, gForceAY, gForceAZ;
float vibration;

// -------- GPS --------
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

// -------- SENSORS --------
int soilPin = 34;
#define RAIN_ANALOG 33
int dryValue = 3500;
int wetValue  = 1500;

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n WiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  gpsSerial.begin(9600, SERIAL_8N1, 16, 17);
  Serial.println("GPS Initialized");

  wakeSensor(MPU_ADDRESS);
  pinMode(RAIN_ANALOG, INPUT);
}

void loop() {
  // -------- SOIL MOISTURE --------
  int rawADC = analogRead(soilPin);
  float moisturePercent = map(rawADC, dryValue, wetValue, 0, 100);
  moisturePercent = constrain(moisturePercent, 0, 100);
  Serial.print("Moisture: "); Serial.print(moisturePercent); Serial.println("%");

  // -------- RAIN SENSOR --------
  int analogValue = analogRead(RAIN_ANALOG);
  float rainPercent = ((4095.0 - analogValue) / 4095.0) * 100.0;
  Serial.print("Rain: "); Serial.print(rainPercent); Serial.println("%");

  // -------- MPU6050 --------
  readAccelData(MPU_ADDRESS, rawAX, rawAY, rawAZ);
  rawAccelToGForce(rawAX, rawAY, rawAZ, gForceAX, gForceAY, gForceAZ);
  vibration = abs(sqrt(gForceAX * gForceAX +
                       gForceAY * gForceAY +
                       gForceAZ * gForceAZ));
  Serial.print("Vibration: "); Serial.println(vibration);

  // -------- GPS --------
  while (gpsSerial.available()) gps.encode(gpsSerial.read());

  float lat = 0.0, lon = 0.0, altitude = 0.0;
  int satellites = 0;
  if (gps.location.isValid()) {
    lat        = gps.location.lat();
    lon        = gps.location.lng();
    altitude   = gps.altitude.meters();
    satellites = gps.satellites.value();
    Serial.print("Lat: "); Serial.println(lat, 6);
    Serial.print("Lon: "); Serial.println(lon, 6);
  }

  // -------- BUILD JSON --------
  StaticJsonDocument<256> doc;
  doc["moisture"]   = moisturePercent;
  doc["rain"]       = rainPercent;
  doc["vibration"]  = vibration;
  doc["lat"]        = lat;
  doc["lon"]        = lon;
  doc["altitude"]   = altitude;
  doc["satellites"] = satellites;

  char payload[256];
  serializeJson(doc, payload);

  // -------- HTTP POST --------
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    int responseCode = http.POST(payload);

    if (responseCode > 0) {
      Serial.println(" Sent: " + String(payload));
      Serial.println("Server: " + http.getString());
    } else {
      Serial.printf("HTTP failed: %s\n", http.errorToString(responseCode).c_str());
    }
    http.end();
  } else {
    Serial.println(" WiFi disconnected — reconnecting...");
    WiFi.reconnect();
    delay(3000);
  }

  Serial.println("-------------------------");
  delay(2000);
}
