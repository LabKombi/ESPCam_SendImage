#define MAXBUFFERSIZE 12288

#include <WiFi.h>
#include <LittleFS.h>
#include <esp_camera.h>
#include <Adafruit_MQTT.h>
#include <Adafruit_MQTT_Client.h>
#include <mbedtls/base64.h>
#include <pin.h>

#define WIFI_SSID " "
#define WIFI_PASSWORD "bintang8kali"

#define AIO_SERVER "io.adafruit.com"
#define AIO_SERVERPORT 1883
#define AIO_USERNAME "-"
#define AIO_KEY "-"
#define FILE_PHOTO_PATH "/photo.jpg"

WiFiClient client;
Adafruit_MQTT_Client mqtt(&client, AIO_SERVER, AIO_SERVERPORT, AIO_USERNAME, AIO_KEY);
Adafruit_MQTT_Publish photoFeed = Adafruit_MQTT_Publish(&mqtt, AIO_USERNAME "/feeds/photo");

void capturePhotoSaveLittleFS() {
  camera_fb_t* fb = NULL;

  // Skip initial frames
  for (int i = 0; i < 4; i++) {
    fb = esp_camera_fb_get();
    esp_camera_fb_return(fb);
  }

  // Capture photo
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  File file = LittleFS.open(FILE_PHOTO_PATH, FILE_WRITE);
  if (!file) {
    Serial.println("Failed to open file for writing");
  } else {
    file.write(fb->buf, fb->len);
    Serial.println("Photo saved to LittleFS");
  }
  file.close();
  esp_camera_fb_return(fb);
}

bool publishBase64(const uint8_t* data, size_t len) {
  size_t out_len = 4 * ((len + 2) / 3);
  uint8_t* out = (uint8_t*)ps_malloc(out_len + 1);
  if (!out) return false;

  if (mbedtls_base64_encode(out, out_len + 1, &out_len, data, len) != 0) {
    free(out);
    return false;
  }
  out[out_len] = '\0';

  // Mix header dan data
  String base64_str = "data:image/jpeg;base64,";
  base64_str += (char*)out;

  Serial.print("Sending Total Size: ");
  Serial.println(base64_str.length());

  // Send base64_str, not out
  bool ok = photoFeed.publish(base64_str.c_str()); 
  
  free(out);
  return ok;
}

void sendPhotoToAdafruitIO() {
  File file = LittleFS.open(FILE_PHOTO_PATH, "r");
  if (!file) {
    Serial.println("Photo not found in LittleFS");
    return;
  }

  if (!mqtt.connected()) {
    reconnectMQTT();
  }

  size_t file_size = file.size();
  uint8_t* img = (uint8_t*)ps_malloc(file_size);
  if (!img) {
    Serial.println("Image buffer alloc failed");
    file.close();
    return;
  }

  file.read(img, file_size);
  file.close();

  if (publishBase64(img, file_size)) {
    Serial.println("Photo sent to Adafruit IO");
  } else {
    Serial.println("Failed to send photo data to Adafruit IO");
  }

  free(img);
}

void reconnectMQTT() {
  int8_t ret = mqtt.connect();
  while (ret != 0) {
    Serial.print("MQTT connection failed, error code: ");
    Serial.println(mqtt.connectErrorString(ret));
    mqtt.disconnect();
    delay(5000);
    ret = mqtt.connect();
  }

  Serial.println("MQTT connected!");
}

void setup() {
  Serial.begin(9600);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.println("Connecting to WiFi...");
    Serial.println(WIFI_SSID);
    delay(1000);
  }
  Serial.println("WiFi connected");
  Serial.println(WiFi.localIP());

  if (!LittleFS.begin(true)) {
    Serial.println("Failed to mount LittleFS");
    return;
  }

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
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
  }

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera initialization failed");
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  s->set_brightness(s, 1);     // Brighten the image (-2 to 2)
  s->set_contrast(s, 1);       // Increase contrast
  s->set_saturation(s, 0);     // Normal saturation setting
  s->set_whitebal(s, 1);       // Auto set white balance
  s->set_awb_gain(s, 1);       // Automatic White Balance gain
  s->set_exposure_ctrl(s, 1);  // Automatic exposure control
}

void loop() {
  capturePhotoSaveLittleFS();
  sendPhotoToAdafruitIO();
  delay(10000); // Delay 10 seconds between uploads
}
