from imports import *
from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED
# from flask import Flask, request, jsonify # type: ignore
import numpy as np
import cv2  # type: ignore
import os
import io
import base64
from PIL import Image
from tensorflow.keras.models import load_model  # type: ignore
from paho.mqtt.client import Client # type: ignore
import time

app = Flask(__name__)

# Adafruit IO Configuration
print("Server:", AIO_SERVER)
print("Username:", AIO_USERNAME)
print("Feed:", AIO_FEED)

# Folder untuk menyimpan foto yang diterima
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Model dan Konfigurasi
model = load_model("model/cnn_model_mobilenet.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EYE_CLOSED_THRESHOLD = 1.5  # ambang waktu mata tertutup
YAWN_THRESHOLD = 3.0  # Ambang waktu menguap
FRAME_COUNT = 0  
last_eye_status = "open"
last_yawn_status = "no_yawn"
start_time = None

# Fungsi Preprocessing
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (64, 64))
    face_normalized = face_resized / 255.0
    face_final = np.expand_dims(face_normalized, axis=-1)
    face_final = np.expand_dims(face_final, axis=0)
    return face_final

    # Simpan gambar preprocessed
    # cv2.imwrite(os.path.join(UPLOAD_FOLDER, "proces_image.jpg"), face_normalized)

# Callback MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(AIO_FEED)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    print(f"Received message from MQTT feed: {msg.topic}")
    try:
        # Decode image
        img_data = base64.b64decode(msg.payload)
        image = Image.open(io.BytesIO(img_data))
        image = np.array(image)

        if image is None or len(image.shape) != 3:
            print("Invalid image received.")
            return

        # Preprocess and predict
        processed_image = preprocess_image(image)
        if processed_image is None:
            print("No face detected in the image.")
            return

        # Save for debugging
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "mqtt_image.jpg"), image)   # menyimpan gambar asli

        prediction = model.predict(processed_image)
        eye_status = "closed" if prediction[0][0] > 0.5 else "open"
        print(f"Eye status: {eye_status}")



    except Exception as e:
        print(f"Error processing image: {e}")

# Endpoint untuk review gambar yang telah diproses
# @app.route('/review', methods=['GET'])
# def review():
#     processed_image_path = os.path.join(UPLOAD_FOLDER, "mqtt_image.jpg")
#     if os.path.exists(processed_image_path):
#         return send_file(processed_image_path, mimetype='image/jpeg')
#     else:
#         return jsonify({"error": "No processed image found"}), 404

# MQTT Client Setup
mqtt_client = Client()
mqtt_client.username_pw_set(AIO_USERNAME, AIO_KEY)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(AIO_SERVER, 1883)
mqtt_client.loop_start()

@app.route('/')
def index():
    return "Server is running and connected to MQTT feed."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
