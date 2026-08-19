#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"

// ===============================
// WiFi
// ===============================
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";

// Replace with your PC's IP address
const char* serverURL = "http://ipaddress:8000/predict";

// Capture every 5 seconds
const unsigned long interval = 5000;
unsigned long lastCapture = 0;

// ===============================
// AI-THINKER ESP32-CAM pins
// ===============================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22


// ===============================
// Setup camera
// ===============================
void setupCamera()
{
    camera_config_t config;

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;

    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;

    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;

    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;

    config.xclk_freq_hz = 20000000;

    config.pixel_format = PIXFORMAT_JPEG;

    // Image size
    config.frame_size = FRAMESIZE_VGA;   // 640x480
    config.jpeg_quality = 10;
    config.fb_count = 2;

    esp_err_t result = esp_camera_init(&config);

    if (result != ESP_OK)
    {
        Serial.print("Camera initialization failed: 0x");
        Serial.println(result, HEX);

        while (true)
        {
            delay(1000);
        }
    }

    Serial.println("Camera initialized.");
}


// ===============================
// Connect WiFi
// ===============================
void connectWiFi()
{
    WiFi.begin(ssid, password);

    Serial.print("Connecting to WiFi");

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("WiFi connected!");

    Serial.print("ESP32-CAM IP: ");
    Serial.println(WiFi.localIP());
}

// ===============================
// Capture and send image
// ===============================
void captureAndSend()
{
    Serial.println("Capturing image...");

    camera_fb_t* fb = esp_camera_fb_get();

    if (fb == NULL)
    {
        Serial.println("Camera capture failed!");
        return;
    }

    Serial.print("Image size: ");
    Serial.print(fb->len);
    Serial.println(" bytes");

    HTTPClient http;
    http.begin(serverURL);
    
    http.addHeader("Content-Type", "image/jpeg");
    Serial.println("Sending image...");

    int responseCode = http.POST(
        fb->buf,
        fb->len
    );

    if (responseCode > 0)
    {
        Serial.print("Server response: ");
        Serial.println(responseCode);
        String response = http.getString();
        Serial.println(response);
    }
    else
    {
        Serial.print("Send failed: ");
        Serial.println(
            http.errorToString(responseCode)
        );
    }

    http.end();

    // Return camera buffer
    esp_camera_fb_return(fb);

    Serial.println("Done.");
}


// ===============================
// Setup
// ===============================
void setup()
{
    Serial.begin(115200);

    delay(1000);

    Serial.println();
    Serial.println("ESP32-CAM Image Sender");

    setupCamera();
    connectWiFi();
}


// ===============================
// Loop
// ===============================
void loop()
{
    if (millis() - lastCapture >= interval)
    {
        lastCapture = millis();

        captureAndSend();
    }
}