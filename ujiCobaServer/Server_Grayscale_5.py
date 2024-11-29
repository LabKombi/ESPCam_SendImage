import os
import time
import base64
import numpy as np
import io
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import dlib
from imutils import face_utils
from paho.mqtt.client import Client
from PIL import Image
from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

model = load_model("model/cnn_model_mobilenet.h5")

landmark_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

app = Flask(__name__)

CLASSES = ["not_sleepy", "sleepy", "yawn"]

EYE_CLOSED_THRESHOLD = 1.5
YAWN_THRESHOLD = 3.0

eye_closed_start_time = None
yawn_start_time = None

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Preprocessing function
def preprocess_image(image, target_size=(128, 128), feature="eye"):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_image, 1)

    if len(faces) == 0:
        raise ValueError("No face detected in the image")

    for face in faces:
        landmarks = landmark_predictor(gray_image, face)
        landmarks = face_utils.shape_to_np(landmarks)

        if feature == "eye":
            eye_points = np.concatenate([landmarks[36:42], landmarks[42:48]])
            x_min, y_min = eye_points.min(axis=0)
            x_max, y_max = eye_points.max(axis=0)
        elif feature == "mouth":
            mouth_points = landmarks[48:68]
            x_min, y_min = mouth_points.min(axis=0)
            x_max, y_max = mouth_points.max(axis=0)
        else:
            raise ValueError("Invalid feature type")

        margin = 15
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(gray_image.shape[1], x_max + margin)
        y_max = min(gray_image.shape[0], y_max + margin)

        region = gray_image[y_min:y_max, x_min:x_max]
        region_resized = cv2.resize(region, target_size, interpolation=cv2.INTER_CUBIC)
        region_normalized = region_resized / 255.0
        return np.expand_dims(region_normalized, axis=(0, -1))

    raise ValueError("No valid region detected")

def process_and_predict(image):
    global eye_closed_start_time, yawn_start_time

    result = "not_sleepy"
    confidence = 0.0

    try:
        # Eye prediction
        preprocessed_eye_image = preprocess_image(image, feature="eye")
        predictions = model.predict(preprocessed_eye_image)
        predicted_class_eye = CLASSES[np.argmax(predictions)]
        confidence_eye = np.max(predictions)

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
        preprocessed_mouth_image = preprocess_image(image, feature="mouth")
        predictions = model.predict(preprocessed_mouth_image)
        predicted_class_mouth = CLASSES[np.argmax(predictions)]
        confidence_mouth = np.max(predictions)

        if predicted_class_mouth == "yawn":
            if yawn_start_time is None:
                yawn_start_time = current_time
            elif current_time - yawn_start_time >= YAWN_THRESHOLD:
                result = "yawn"
                yawn_start_time = None
        else:
            yawn_start_time = None

        confidence = max(confidence_eye, confidence_mouth)

    except ValueError as e:
        print(f"Prediction error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return {"result": result, "confidence": confidence}

# MQTT Callbacks
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code " + str(rc))
    client.subscribe("your/mqtt/topic")

def on_message(client, userdata, msg):
    print(f"Received message from MQTT feed: {msg.topic}")
    try:
        # Decode base64 image
        img_data = base64.b64decode(msg.payload)
        image = Image.open(io.BytesIO(img_data))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Save received image for debugging
        processed_image_path = os.path.join(UPLOAD_FOLDER, "mqtt_image.jpg")
        cv2.imwrite(processed_image_path, image)

        # Perform prediction
        prediction = process_and_predict(image)
        print(f"MQTT Prediction: {prediction}")

    except Exception as e:
        print(f"Error processing MQTT image: {e}")

# MQTT Setup
AIO_SERVER = "mqtt.example.com"
AIO_USERNAME = "your_username"
AIO_KEY = "your_password"

mqtt_client = Client()
mqtt_client.username_pw_set(AIO_USERNAME, AIO_KEY)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Start MQTT loop
print("Connecting to MQTT server:", AIO_SERVER)
mqtt_client.connect(AIO_SERVER, 1883)
mqtt_client.loop_start()

# Main program
if __name__ == "__main__":
    print("MQTT-based Sleep Detection Service is running...")
    while True:
        time.sleep(1)