import os
import time
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import dlib
from imutils import face_utils

model = load_model("model/mobilenet_rgb.h5")
# model = load_model("model/new_model_mobilenet.h5")

landmark_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

app = Flask(__name__)

CLASSES = ["not_sleepy", "sleepy", "yawn"]

EYE_CLOSED_THRESHOLD = 2.0
YAWN_THRESHOLD = 4.0

eye_closed_start_time = None
yawn_start_time = None

def preprocess_image(image_path, target_size=(64, 64), feature="eye"):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unreadable")

    # 2. Menghilangkan noise menggunakan Gaussian Blur
    gray_blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 3. Menajamkan gambar
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray_sharpened = cv2.filter2D(gray_blurred, -1, kernel_sharpening)

    faces = face_detector(gray_sharpened, 1)
    if len(faces) == 0:
        raise ValueError("No face detected in the image")

    for face in faces:

        landmarks = landmark_predictor(image, face)
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
        x_max = min(image.shape[1], x_max + margin)
        y_max = min(image.shape[0], y_max + margin)

        region = image[y_min:y_max, x_min:x_max]
        # region_resized = cv2.resize(region, target_size, interpolation=cv2.INTER_CUBIC)
        # region_normalized = region_resized / 255.0
        region_rgb = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
        region_resized = cv2.resize(region_rgb, target_size, interpolation=cv2.INTER_CUBIC)
        region_normalized = region_resized / 255.0
        return np.expand_dims(region_normalized, axis=(0, -1))

    raise ValueError("No valid region detected")

@app.route('/predict', methods=['POST'])
def predict():
    global eye_closed_start_time, yawn_start_time

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    result = "not_sleepy"

    try:

        preprocessed_eye_image = preprocess_image(file_path, feature="eye")

        predictions = model.predict(preprocessed_eye_image)
        predicted_class_eye = CLASSES[np.argmax(predictions)]

        current_time = time.time()
        if predicted_class_eye == "sleepy":
            if eye_closed_start_time is None:
                eye_closed_start_time = current_time
            elif current_time - eye_closed_start_time >= EYE_CLOSED_THRESHOLD:
                result = "sleepy"
                eye_closed_start_time = None
        else:
            eye_closed_start_time = None

        preprocessed_mouth_image = preprocess_image(file_path, feature="mouth")

        predictions = model.predict(preprocessed_mouth_image)
        predicted_class_mouth = CLASSES[np.argmax(predictions)]

        if predicted_class_mouth == "yawn":
            if yawn_start_time is None:
                yawn_start_time = current_time
            elif current_time - yawn_start_time >= YAWN_THRESHOLD:
                result = "yawn"
                yawn_start_time = None
        else:
            yawn_start_time = None

    except ValueError as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": "Internal server error"}), 500

    os.remove(file_path)

    return jsonify({"prediction": result, "confidence": float(np.max(predictions))})

if __name__ == '__main__':
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)