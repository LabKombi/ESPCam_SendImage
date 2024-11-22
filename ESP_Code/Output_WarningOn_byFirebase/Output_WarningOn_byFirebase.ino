#include <ESP8266WiFi.h>
#include <Firebase_ESP_Client.h>
#include <SPI.h>

FirebaseData firebaseData;
FirebaseAuth auth;
FirebaseConfig config;

#define WIFI_SSID " "
#define WIFI_PASSWORD "bintang8kali"

#define FIREBASE_HOST "led-switch-6c690-default-rtdb.firebaseio.com"
#define FIREBASE_AUTH "dOtE0PZd56lmLBJE7NL0klwOafm5t254kS7uqQRx"

#define PIN 2 // D4

void setup() {
  Serial.begin(115200);
  pinMode(PIN, OUTPUT);
  digitalWrite(PIN, LOW);

  SPI.begin();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(WIFI_SSID);
  Serial.println("Connected to Wi-Fi");

  config.host = FIREBASE_HOST;
  config.signer.tokens.legacy_token = FIREBASE_AUTH;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

void loop() {
  // Ambil status dari database Firebase
  if (Firebase.RTDB.getString(&firebaseData, "/FBStatus")) { // Path ke data di Firebase
    if (firebaseData.dataType() == "string") {
      String FBStatus = firebaseData.stringData();
      
      if (FBStatus == "ON") {
        Serial.println("Relay ON");
        digitalWrite(PIN, HIGH);
      } 
      else if (FBStatus == "OFF") {
        Serial.println("Relay OFF");
        digitalWrite(PIN, LOW);
      } 
      else {
        Serial.println("Salah kode! Isi dengan data ON/OFF.");
      }
    }
  } else {
    Serial.println("Gagal mendapatkan data dari Firebase");
    Serial.println(firebaseData.errorReason()); // Tampilkan alasan error
  }

  delay(5000);
}
