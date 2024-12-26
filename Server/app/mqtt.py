import base64
import os
import cv2
import paho.mqtt.client as mqtt
from predictor import predict

from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

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

        # Decode pesan dari base64
        image_data = msg.payload.decode("utf-8")
        image_bytes = base64.b64decode(image_data)
        
        # Simpan gambar ke file sementara
        temp_image_path = os.path.join(TEMP_DIR, "received_image.jpg")
        with open(temp_image_path, "wb") as image_file:
            image_file.write(image_bytes)

        # Load gambar ke dalam format numpy array
        image_np = cv2.imread(temp_image_path)

        if image_np is None:
            raise ValueError("Failed to load image from temporary file.")

        # Jalankan prediksi
        result, confidence = predict(image_np)  # Kirim numpy array, bukan path
        print(f"Prediction result: {result}, Confidence: {confidence}")

        # Hapus file sementara
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

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