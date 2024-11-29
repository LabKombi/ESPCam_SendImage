#include <Adafruit_MQTT.h>
#include <Adafruit_MQTT_Client.h>
#include <WiFi.h>
#include "esp_camera.h"
#include <base64.h>
#include <SPIFFS.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "driver/rtc_io.h"
#include <FS.h>
#include <ArduinoJson.h>

// pin cam
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define cols 320
#define rows 240
#define size 3

// WiFi
#define WLAN_SSID "-" // isi dengan nama Wi-Fi
#define WLAN_PASS "-" // isi dengan password Wi-Fi

// #define WLAN_SSID "Your-SSID-WiFi" // isi dengan nama Wi-Fi
// #define WLAN_PASS "Your-Password-WiFi" // isi dengan password Wi-Fi

// Adafruit
#define AIO_SERVER "io.adafruit.com"
#define AIO_SERVERPORT 1883
#define AIO_USER "-" // isi dengan Adafruit IO username
#define AIO_KEY "-" // isi dengan Adafruit IO Key

WiFiClient client;

Adafruit_MQTT_Client mqtt(&client, AIO_SERVER, AIO_SERVERPORT, AIO_USER, AIO_KEY);
Adafruit_MQTT_Publish camera = Adafruit_MQTT_Publish(&mqtt, AIO_USER "/feeds/photo");

static bool is_initialised = false;
uint8_t *snapshot_buf;
unsigned long skr = 0;
static camera_config_t camera_config = {
  .pin_pwdn = PWDN_GPIO_NUM,
  .pin_reset = RESET_GPIO_NUM,
  .pin_xclk = XCLK_GPIO_NUM,
  .pin_sscb_sda = SIOD_GPIO_NUM,
  .pin_sscb_scl = SIOC_GPIO_NUM,
  .pin_d7 = Y9_GPIO_NUM,
  .pin_d6 = Y8_GPIO_NUM,
  .pin_d5 = Y7_GPIO_NUM,
  .pin_d4 = Y6_GPIO_NUM,
  .pin_d3 = Y5_GPIO_NUM,
  .pin_d2 = Y4_GPIO_NUM,
  .pin_d1 = Y3_GPIO_NUM,
  .pin_d0 = Y2_GPIO_NUM,
  .pin_vsync = VSYNC_GPIO_NUM,
  .pin_href = HREF_GPIO_NUM,
  .pin_pclk = PCLK_GPIO_NUM,
  .xclk_freq_hz = 20000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,
  .pixel_format = PIXFORMAT_JPEG,
  .frame_size = FRAMESIZE_QVGA,
  .jpeg_quality = 12,
  .fb_count = 1,
  .fb_location = CAMERA_FB_IN_PSRAM,
  .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

bool ainit(void);
void deinit(void);
bool capture(uint8_t *out_buf);

void setup() {
  Serial.begin(115200);
  if (ainit() == false) {
    Serial.println("Failed to initialize Camera!");
  } else {
    Serial.println("Camera initialized");
  }
  pinMode(4, OUTPUT);
  digitalWrite(4, LOW);
  setup_wifi();
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
}

// void loop() {
//   if (millis() - skr > 1000) {
//     skr = millis();
//     MQTT_connect();
//     snapshot_buf = (uint8_t *)malloc(cols * rows * size);
//     if (snapshot_buf == nullptr) {
//       Serial.println("ERR: Failed to allocate snapshot buffer!");
//       return;
//     }
//     if (capture(snapshot_buf) == false) {
//       Serial.println("Failed to capture image");
//       return;
//     }
//     free(snapshot_buf);
//   }
// }

// unsigned long skr = 0; // Menyimpan waktu terakhir
int iterationCount = 0; // Menghitung jumlah iterasi

void loop() {
  // Mengecek apakah 1 detik telah berlalu
  if (millis() - skr > 1000) {
    skr = millis(); // Memperbarui waktu terakhir
    iterationCount++; // Meningkatkan hitungan iterasi

    // Koneksi MQTT
    MQTT_connect();

    // Alokasi memori untuk buffer gambar
    snapshot_buf = (uint8_t *)malloc(cols * rows * size);
    if (snapshot_buf == nullptr) {
      Serial.println("ERR: Failed to allocate snapshot buffer!");
      return;
    }

    // Pengambilan gambar
    if (capture(snapshot_buf) == false) {
      Serial.println("Failed to capture image");
      free(snapshot_buf); // Pastikan memori dibebaskan jika gagal
      return;
    }

    // Membebaskan buffer setelah digunakan
    free(snapshot_buf);

    // Mengecek apakah iterasi telah mencapai 30 kali
    if (iterationCount >= 30) {
      Serial.println("Pausing for 30 seconds...");
      delay(30000); // Delay selama 30 detik
      iterationCount = 0; // Reset penghitung iterasi
    }
  }
}


// Connect to Wifi
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(WLAN_SSID);
  WiFi.begin(WLAN_SSID, WLAN_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

// Inisiasi Camera
bool ainit(void) {
  if (is_initialised) return true;
  esp_err_t err = esp_camera_init(&camera_config);
  if (err != ESP_OK) {
    Serial.println("Camera init failed with error 0x%x\n");
    return false;
  }
  is_initialised = true;
  return true;
}

void deinit(void) {
  esp_err_t err = esp_camera_deinit();
  if (err != ESP_OK) {
    Serial.println("Camera deinit failed\n");
    return;
  }
  is_initialised = false;
  return;
}

// Capture picture
bool capture(uint8_t *out_buf) {
  if (!is_initialised) {
    Serial.println("ERR: Camera is not initialized");
    return false;
  }
  digitalWrite(4, HIGH);  // flash on
  camera_fb_t *fb = NULL;
  for (int i = 0; i < 4; i++) {
    fb = esp_camera_fb_get();
    esp_camera_fb_return(fb);
    fb = NULL;
  }
  fb = NULL;
  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }
  String buffer = base64::encode((uint8_t *)fb->buf, fb->len);
  unsigned int length();
  Serial.println("Buffer Length: ");
  Serial.print(buffer.length());
  Serial.println("");
  Serial.println("Publishing...");
  if (!camera.publish(buffer.c_str())) Serial.println(F("failed"));
  else Serial.println(F("worked"));
  esp_camera_fb_return(fb);
  digitalWrite(4, LOW);  // flash off
  return true;
}

// Connect to MQTT
void MQTT_connect() {
  int8_t ret;
  if (mqtt.connected()) {
    return;
  }
  Serial.print("Connecting to MQTT... ");
  uint8_t retries = 3;
  while ((ret = mqtt.connect()) != 0) { 
    Serial.println(mqtt.connectErrorString(ret));
    Serial.println("Retrying MQTT connection in 5 seconds...");
    mqtt.disconnect();
    delay(5000);
    retries--;
    if (retries == 0) {
      while (1)
        ;
    }
  }
  Serial.println("MQTT Connected!");
}