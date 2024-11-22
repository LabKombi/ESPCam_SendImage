#include <WiFi.h>
#include <LittleFS.h>
#include <esp_camera.h>
#include <WebServer.h>
#include <pin.h>

#define WIFI_SSID " "
#define WIFI_PASSWORD "bintang8kali"

// #define WIFI_SSID "JTI-3.02"
// #define WIFI_PASSWORD ""

#define FILE_PHOTO_PATH "/photo.jpg"

WebServer server(80);

void capturePhotoSaveLittleFS() {
  camera_fb_t* fb = NULL;
  
  // Skip the first few frames to get a clear picture
  for (int i = 0; i < 4; i++) {
    fb = esp_camera_fb_get();
    esp_camera_fb_return(fb);
    fb = NULL;
  }
  
  // Take a new photo
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  File file = LittleFS.open(FILE_PHOTO_PATH, FILE_WRITE);
  if (!file) {
    Serial.println("Failed to open file in writing mode");
  } else {
    file.write(fb->buf, fb->len);
    Serial.println("Photo saved to LittleFS");
  }
  file.close();
  esp_camera_fb_return(fb);
}
void handlePhoto() {
  File file = LittleFS.open(FILE_PHOTO_PATH, "r");
  if (!file) {
    server.send(404, "text/plain", "Photo not found");
    return;
  }
  server.streamFile(file, "image/jpeg");
  file.close();
}

void setup() {
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.println("Connecting to WiFi...");
  }
  Serial.print("Connected: ");
  Serial.println(WIFI_SSID);
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  if (!LittleFS.begin(true)) {
    Serial.println("An error occurred while mounting LittleFS");
    return;
  }

  // Set up camera
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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 1;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    return;
  }

  // Route to capture and serve the photo
  server.on("/photo.jpg", HTTP_GET, handlePhoto);
  server.begin();
}

void loop() {
  server.handleClient();
  capturePhotoSaveLittleFS();
}
