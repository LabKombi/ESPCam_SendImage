import base64
import os
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from predictor import predict

try:
    import winsound
except ImportError:
    winsound = None

from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

TEMP_DIR = "temp"
last_alarm_status = None
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


def play_alarm(status):
    global last_alarm_status

    if status in ("sleepy", "yawn") and status != last_alarm_status:
        if winsound is not None:
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:
            print("Alarm Windows tidak tersedia pada sistem ini")
        print(f"ALARM: {status}")

    last_alarm_status = status if status in ("sleepy", "yawn") else None

# Callback ketika terhubung ke MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to Adafruit IO!")
        client.subscribe(AIO_FEED)
    else:
        print(f"Failed to connect, return code {rc}")

# Callback ketika menerima pesan dari feed
def on_message(client, userdata, msg):
    try:
        print(f"Message received on {msg.topic}")

        # 1. Get data string
        payload = msg.payload.decode("utf-8")

        if "," in payload:
            payload = payload.split(",")[1]

        # 3. Fix padding: Add '=' so the panjangnya kelipatan 4
        missing_padding = len(payload) % 4
        if missing_padding:
            payload += '=' * (4 - missing_padding)
        
        # 4. Decode to bytes
        image_bytes = base64.b64decode(payload)
        
        # 5. Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_np is None:
            print("Failed to decode image")
            return
        
        # Save image for debugging
        cv2.imwrite("debug_foto_esp32.jpg", image_np)

        # 6. DETEKSI
        result, confidence = predict(image_np) 
        print(f"Hasil Deteksi: {result} | Confidence: {confidence}")
        play_alarm(result)

    except Exception as e:
        print(f"Error processing message: {e}")

# Inisialisasi MQTT client
client = mqtt.Client()
client.username_pw_set(AIO_USERNAME, AIO_KEY)
client.on_connect = on_connect
client.on_message = on_message

# Hubungkan ke broker
client.connect(AIO_SERVER, 1883, 60)

# Mulai loop
print("Listening for messages...")
client.loop_forever()