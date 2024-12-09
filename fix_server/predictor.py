import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import dlib

from firebase_utils import initialize_firebase, send_status_to_firebase
from image_processing import preprocess_image

# Load model
model = load_model("model/mobilenet_rgb_4.h5")
model.make_predict_function()  # Pastikan model dalam mode evaluasi

app = Flask(__name__)   # inisialisi flask server

# kelas target
CLASSES = ["closed", "open", "no_yawn", "yawn"]

result = None  # global threshold

initialize_firebase()   # inisialisasi firebase

def predict(image_path):
    global result

    try:
        preprocessed_eye_l_image = preprocess_image(image_path, feature="eye_l")
        preprocessed_eye_r_image = preprocess_image(image_path, feature="eye_r")
        preprocessed_mouth_image = preprocess_image(image_path, feature="mouth")

        # Proses prediksi mata kiri
        predictions_eye_l = model.predict(preprocessed_eye_l_image)
        predicted_class_eye_l = CLASSES[np.argmax(predictions_eye_l)]
        eye_l_confidence = np.max(predictions_eye_l)

        # Proses prediksi mata kanan
        predictions_eye_r = model.predict(preprocessed_eye_r_image)
        predicted_class_eye_r = CLASSES[np.argmax(predictions_eye_r)]
        eye_r_confidence = np.max(predictions_eye_r)

        # Proses prediksi mulut
        predictions_mouth = model.predict(preprocessed_mouth_image)
        predicted_class_mouth = CLASSES[np.argmax(predictions_mouth)]
        mouth_confidence = np.max(predictions_mouth)

        print(f"{result} - {predicted_class_eye_l} - {eye_l_confidence}, {predicted_class_eye_r} - {eye_r_confidence}, {predicted_class_mouth} - {mouth_confidence}")
        
        print(f"eye_l : {predicted_class_eye_l} - {predictions_eye_l}")
        print(f"eye_r : {predicted_class_eye_r} - {predictions_eye_r}")
        print(f"mouth : {predicted_class_mouth} - {predictions_mouth}")

        # Ekstrak nilai berdasarkan indeks
        eye_l_closed = predictions_eye_l[0, 0]
        eye_l_open = predictions_eye_l[0, 1]
        eye_r_closed = predictions_eye_l[0, 0]
        eye_r_open = predictions_eye_l[0, 1]
        mouth_no_yawn = predictions_mouth[0, 2]
        mouth_yawn = predictions_mouth[0, 3]

        print(f"eye_l - closed : {eye_l_closed} - open : {eye_l_open}")
        print(f"eye_r - closed : {eye_r_closed} - open : {eye_r_open}")
        print(f"mouth - no_yawn : {mouth_no_yawn} - yawn : {mouth_yawn}")

        sleepy = ( eye_l_closed + eye_r_closed ) / 2
        no_sleepy = ( eye_l_closed + eye_r_closed ) / 2

        print(f"sleepy : {sleepy}")
        print(f"no_sleepy : {no_sleepy}")

        confidence = max(sleepy, mouth_yawn)

        if (
            mouth_yawn > 0.6
            # or predicted_class_mouth == 'open'
            # or predicted_class_mouth == 'yawn'
            ):
                result = "yawn"
                send_status_to_firebase(result, confidence)
                return "yawn"
        elif (
            eye_l_open > 0.5
            or eye_r_open > 0.5
            or predicted_class_eye_l == 'yawn'
            or predicted_class_eye_r =='yawn'
            or predicted_class_eye_l == 'open'
            or predicted_class_eye_r == 'open'
            ):  
                result = "not_sleepy"
                # send_status_to_firebase(result, confidence)
                return "not_sleepy"
        elif ( 
            eye_l_closed > 0.5
            and eye_r_closed > 0.5
            or predicted_class_eye_l == 'no_yawn'
            or predicted_class_eye_r == 'no_yawn'
            or predicted_class_eye_l == 'closed'
            or predicted_class_eye_r == 'closed'
            ):
                result = "sleepy"
                send_status_to_firebase(result, confidence)
                return "sleepy"

        return result

    except Exception as e:
        print(f"Prediction error: {e}")
        return "error"