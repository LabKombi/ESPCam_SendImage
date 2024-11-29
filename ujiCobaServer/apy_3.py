import os
import time
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import dlib
from imutils import face_utils

import firebase_admin
from firebase_admin import credentials, db

# Load model
model = load_model("model/mobilenet_rgb_4.h5")
model.make_predict_function()  # Ensure the model is in evaluation mode

# Load landmark predictor
landmark_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

app = Flask(__name__)

CLASSES = ["closed", "open", "no_yawn", "yawn"]

# EYE_CLOSED_THRESHOLD = 1.0
# YAWN_THRESHOLD = 2.0

# eye_closed_start_time = None
# yawn_start_time = None

# Buffer counters for yawn and closed
# yawn_count = 0
# sleepy_count = 0
# not_count = 0

# Threshold for buffer
# SLEEPY_BUFFER_THRESHOLD = 1
# YAWN_BUFFER_THRESHOLD = 1
# NOT_BUFFER_THRESHOLD = 1

# === Firebase Configuration ===
cred = credentials.Certificate("config/drowsines-key-firebase.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://drowsines-e2e79-default-rtdb.firebaseio.com/"
})

def send_status_to_firebase(status, confidence):
    """
    Mengirimkan status ke Firebase Realtime Database
    """
    ref = db.reference("/pengendara")  # Buat referensi ke node 'status'
    ref.set(status)
    print(f"Status '{status}' berhasil dikirim ke Firebase.")


def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# def remove_noise_gaussian(image, kernel_size=(3, 3)):
    # return cv2.GaussianBlur(image, kernel_size, 0)

def remove_noise_bilateral(image, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def normalize_image(image):
    return image / 255.0

def preprocess_image(image_path, target_size=(224, 224), feature="eye_r"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unreadable")

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if feature == "eye_r":
        # Detect face and extract eye region
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = dlib.get_frontal_face_detector()
        faces = detector(img, 1)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        for face in faces:
            landmarks = landmark_predictor(img, face)
            landmarks = face_utils.shape_to_np(landmarks)
            eye_points = landmarks[36:42]
            # eye_points = np.concatenate([landmarks[36:42], landmarks[42:48]])
            x_min, y_min = eye_points.min(axis=0)
            x_max, y_max = eye_points.max(axis=0)

            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img.shape[1], x_max + margin)
            y_max = min(img.shape[0], y_max + margin)

            cropped_img = img[y_min:y_max, x_min:x_max]
            denoised_img = remove_noise_bilateral(cropped_img)
            sharpened_img = apply_sharpening(denoised_img)
            resized_img = cv2.resize(sharpened_img, target_size, interpolation=cv2.INTER_CUBIC)
            # rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            normalized_img = normalize_image(resized_img)

            # Debugging: Save intermediate results
            cv2.imwrite("debug_eye_r_cropped.jpg", cropped_img)
            cv2.imwrite("debug_eye_r_denoised.jpg", denoised_img)
            cv2.imwrite("debug_eye_r_sharpened.jpg", sharpened_img)
            cv2.imwrite("debug_eye_r_resized.jpg", resized_img)

        return np.expand_dims(normalized_img, axis=0)

    if feature == "eye_l":
        # Detect face and extract eye region
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = dlib.get_frontal_face_detector()
        faces = detector(img, 1)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        for face in faces:
            landmarks = landmark_predictor(img, face)
            landmarks = face_utils.shape_to_np(landmarks)
            eye_points = landmarks[42:48]
            # eye_points = np.concatenate([landmarks[36:42], landmarks[42:48]])
            x_min, y_min = eye_points.min(axis=0)
            x_max, y_max = eye_points.max(axis=0)

            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img.shape[1], x_max + margin)
            y_max = min(img.shape[0], y_max + margin)

            cropped_img = img[y_min:y_max, x_min:x_max]
            denoised_img = remove_noise_bilateral(cropped_img)
            sharpened_img = apply_sharpening(denoised_img)
            resized_img = cv2.resize(sharpened_img, target_size, interpolation=cv2.INTER_CUBIC)
            # rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            normalized_img = normalize_image(resized_img)

            # Debugging: Save intermediate results
            cv2.imwrite("debug_eye_l_cropped.jpg", cropped_img)
            cv2.imwrite("debug_eye_l_denoised.jpg", denoised_img)
            cv2.imwrite("debug_eye_l_sharpened.jpg", sharpened_img)
            cv2.imwrite("debug_eye_l_resized.jpg", resized_img)

        return np.expand_dims(normalized_img, axis=0)

    elif feature == "mouth":
        # Detect face and extract mouth region
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = dlib.get_frontal_face_detector()
        faces = detector(img, 1)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        for face in faces:
            landmarks = landmark_predictor(img, face)
            landmarks = face_utils.shape_to_np(landmarks)
            mouth_points = landmarks[48:68]
            x_min, y_min = mouth_points.min(axis=0)
            x_max, y_max = mouth_points.max(axis=0)

            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img.shape[1], x_max + margin)
            y_max = min(img.shape[0], y_max + margin)

            cropped_img = img[y_min:y_max, x_min:x_max]
            denoised_img = remove_noise_bilateral(cropped_img)
            sharpened_img = apply_sharpening(denoised_img)
            resized_img = cv2.resize(sharpened_img, target_size, interpolation=cv2.INTER_CUBIC)
            # rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            normalized_img = normalize_image(resized_img)

            # Debugging: Save intermediate results
            cv2.imwrite("debug_mouth_cropped.jpg", cropped_img)
            cv2.imwrite("debug_mouth_denoised.jpg", denoised_img)
            cv2.imwrite("debug_mouth_sharpened.jpg", sharpened_img)
            cv2.imwrite("debug_mouth_resized.jpg", resized_img)

            return np.expand_dims(normalized_img, axis=0)

    else:
        raise ValueError("Invalid feature type")

# Variabel global untuk menyimpan hasil prediksi terakhir

last_result = None

@app.route('/predict', methods=['POST'])
def predict():
    global last_result

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)


    try:

        result = "not_sleepy"
        confidence = 0.0  # Default confidence

        # Proses prediksi mata
        preprocessed_eye_l_image = preprocess_image(file_path, feature="eye_l")
        predictions_eye_l = model.predict(preprocessed_eye_l_image)
        predicted_class_eye_l = CLASSES[np.argmax(predictions_eye_l)]
        eye_l_confidence = np.max(predictions_eye_l)
        eye_l_confidence_2 = np.min(predictions_eye_l)

        preprocessed_eye_r_image = preprocess_image(file_path, feature="eye_r")
        predictions_eye_r = model.predict(preprocessed_eye_r_image)
        predicted_class_eye_r = CLASSES[np.argmax(predictions_eye_r)]
        eye_r_confidence = np.max(predictions_eye_r)
        eye_r_confidence_2 = np.min(predictions_eye_r)
        # print(f"Eye Prediction: {predicted_class_eye} with confidence {np.max(predictions_eye)}")

        # Proses prediksi mulut
        preprocessed_mouth_image = preprocess_image(file_path, feature="mouth")
        predictions_mouth = model.predict(preprocessed_mouth_image)
        predicted_class_mouth = CLASSES[np.argmax(predictions_mouth)]
        mouth_confidence = np.max(predictions_mouth)
        mouth_confidence_2 = np.min(predictions_mouth)
        # mouth = predictions_mouth[0, 3]
        # print(f"Mouth Prediction: {predicted_class_mouth} with confidence {np.max(predictions_mouth)}")

        print(f"{last_result} - {predicted_class_eye_l} - {eye_l_confidence}, {predicted_class_eye_r} - {eye_r_confidence}, {predicted_class_mouth} - {mouth_confidence}")
        print(f"eye_l : {predictions_eye_l}")
        print(f"eye_r : {predictions_eye_r}")
        print(f"mouth : {predictions_mouth}")

        # Ekstrak nilai berdasarkan indeks
        eye_l_closed = predictions_eye_l[0, 0]
        eye_l_open = predictions_eye_l[0, 1]
        eye_r_closed = predictions_eye_l[0, 0]
        eye_r_open = predictions_eye_l[0, 1]
        mouth_no_yawn = predictions_mouth[0, 2]
        mouth_yawn = predictions_mouth[0, 3]

        print(f"{eye_l_closed}")
        print(f"{eye_l_open}")
        print(f"{mouth_no_yawn}")
        print(f"{mouth_yawn}")

        if (
            mouth_yawn > 0.65
            ):
                last_result = "yawn"
        elif (
            eye_l_open > 0.5
            and eye_r_open > 0.5
            ):  
                last_result = "not_sleepy"
        elif ( 
            eye_l_closed > 0.5
            and eye_r_closed > 0.5
            ):
                last_result = "sleepy"
        else:
            pass

        # Hanya kirim status jika bukan 'not_sleepy'
        if last_result != "not_sleepy":
            send_status_to_firebase(last_result, confidence)

    except ValueError as e:
        print(f"ValueError: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Unhandled error: {e}")
        return jsonify({"error": "Internal server error"}), 500


    os.remove(file_path)

    # Kembalikan hasil terakhir
    return jsonify({
        "prediction": last_result,
        "eye_confidence": float(eye_l_confidence),
        "eye_confidence_2": float(eye_r_confidence),
        "mouth_confidence": float(mouth_confidence)
})

# Manual Prediction Testing
# def test_model(image_path, feature):
#     try:
#         preprocessed_image = preprocess_image(image_path, feature=feature)
#         predictions = model.predict(preprocessed_image)
#         predicted_class = CLASSES[np.argmax(predictions)]
#         print(f"Prediction: {predicted_class}, Confidence: {np.max(predictions)}")
#     except Exception as e:
#         print(f"Error during prediction: {e}")

if __name__ == '__main__':
    if not os.path.exists("temp"):
        os.makedirs("temp")

    # Example manual test
    # test_model("test_eye.jpg", feature="eye")
    # test_model("test_mouth.jpg", feature="mouth")

    # run debug
    app.run(debug=True)

    # run with ipconfig local
    # app.run(host='0.0.0.0', port=5000)
