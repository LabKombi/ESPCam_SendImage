import os
import time
import base64
import numpy as np
import io
from flask import Flask, send_file, jsonify
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array   # type: ignore
import cv2
import dlib # type: ignore
from imutils import face_utils
from paho.mqtt.client import Client
from PIL import Image
from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

# Folder untuk menyimpan gambar
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model dan konfigurasi
model = load_model("model/cnn_model_mobilenet.h5")  # Grayscale model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

# Konstanta
CLASSES = ["not_sleepy", "sleepy", "yawn"]
EYE_CLOSED_THRESHOLD = 1.5
YAWN_THRESHOLD = 3.0
eye_closed_start_time = None
yawn_start_time = None

# Flask app
app = Flask(__name__)

# Preprocess image
def preprocess_image(image, target_size=(64, 64), feature="eye"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces) == 0:
        return None

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        if feature == "eye":
            points = np.concatenate([landmarks[36:42], landmarks[42:48]])
        elif feature == "mouth":
            points = landmarks[48:68]
        else:
            return None
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        region = gray[max(0, y_min-15):min(y_max+15, gray.shape[0]), max(0, x_min-15):min(x_max+15, gray.shape[1])]
        resized = cv2.resize(region, target_size, interpolation=cv2.INTER_CUBIC)
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=(0, -1))

# Process and predict
def process_and_predict(image):
    global eye_closed_start_time, yawn_start_time
    try:
        result = "not_sleepy"
        confidence = 0.0

        # Eye prediction
        preprocessed_eye = preprocess_image(image, feature="eye")
        if preprocessed_eye is not None:
            predictions = model.predict(preprocessed_eye)
            predicted_class_eye = CLASSES[np.argmax(predictions)]
            confidence = float(np.max(predictions))

            current_time = time.time()
            if predicted_class_eye == "sleepy":
                if eye_closed_start_time is None:
                    eye_closed_start_time = current_time
                elif current_time - eye_closed_start_time >= EYE_CLOSED_THRESHOLD:
                    result = "sleepy"
                    eye_closed_start_time = None
            else:
                eye_closed_start_time = None

        # Mouth prediction
        preprocessed_mouth = preprocess_image(image, feature="mouth")
        if preprocessed_mouth is not None:
            predictions = model.predict(preprocessed_mouth)
            predicted_class_mouth = CLASSES[np.argmax(predictions)]

            current_time = time.time()
            if predicted_class_mouth == "yawn":
                if yawn_start_time is None:
                    yawn_start_time = current_time
                elif current_time - yawn_start_time >= YAWN_THRESHOLD:
                    result = "yawn"
                    yawn_start_time = None
            else:
                yawn_start_time = None

        return {"result": result, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code " + str(rc))
    client.subscribe("your/mqtt/topic")

def on_message(client, userdata, msg):
    print(f"Received message from MQTT feed: {msg.topic}")
    try:
        img_data = base64.b64decode(msg.payload)
        image = Image.open(io.BytesIO(img_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Simpan gambar untuk debugging
        processed_image_path = os.path.join(UPLOAD_FOLDER, "mqtt_image.jpg")
        cv2.imwrite(processed_image_path, image)

        # Prediksi
        result = process_and_predict(image)
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, "after_image.jpg"), image)
        print(f"MQTT Prediction: {result}")
    except Exception as e:
        print(f"Error processing MQTT image: {e}")

# MQTT Setup
mqtt_client = Client()
mqtt_client.username_pw_set(AIO_USERNAME, AIO_KEY)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Start MQTT loop
mqtt_client.connect(AIO_SERVER, 1883)
mqtt_client.loop_start()

@app.route('/')
def index():
    return "Server is running and connected to MQTT feed."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
