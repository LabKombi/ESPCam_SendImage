import os
import sys
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from firebase_utils import initialize_firebase, send_status_to_firebase
from i_proc import preprocess_image

# Load model
model = load_model("data/mobilenet_rgb_4.h5")
model.make_predict_function()  # Pastikan model dalam mode evaluasi

# kelas target
CLASSES = ["closed", "open", "no_yawn", "yawn"]

result = None  # global threshold

initialize_firebase()   # inisialisasi firebase

def predict(frame_path):
    global result

    try:
        # PreProses frame untuk setiap fitur
        preprocessed_eye_l_image = preprocess_image(frame_path, feature="eye_l")
        preprocessed_eye_r_image = preprocess_image(frame_path, feature="eye_r")
        preprocessed_mouth_image = preprocess_image(frame_path, feature="mouth")

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
        
        # Ekstrak nilai berdasarkan indeks
        eye_l_closed = predictions_eye_l[0, 0]
        eye_l_open = predictions_eye_l[0, 1]
        eye_r_closed = predictions_eye_l[0, 0]
        eye_r_open = predictions_eye_l[0, 1]
        mouth_no_yawn = predictions_mouth[0, 2]
        mouth_yawn = predictions_mouth[0, 3]

        sleepy = ( eye_l_closed + eye_r_closed ) / 2
        no_sleepy = ( eye_l_open + eye_r_open ) / 2

        print(f"sleepy : {sleepy}")
        print(f"no_sleepy : {no_sleepy}")
        print(f"yawn : {mouth_yawn}")

        confidence = max(sleepy, no_sleepy, mouth_yawn)

        if (
            sleepy > 0.5
            ):
                result = "sleepy"
                send_status_to_firebase(result, confidence)
        elif ( 
            mouth_yawn > 0.6
            and predicted_class_mouth == 'yawn'
            ):
                result = "yawn"
                send_status_to_firebase(result, confidence)
        elif (
            predicted_class_eye_l == 'open'
            or predicted_class_eye_r == 'open'
            ):  
                result = "not_sleepy"

        return result, float(confidence)

    except Exception as e:
        print(f"Prediction error: {e}")
        return "error", 0.0

if __name__ == "__main__":
    import argparse

    # Parser untuk argumen input
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path ke gambar input")
    args = parser.parse_args()

    # Load gambar
    if not os.path.exists(args.image):
        print(json.dumps({"error": "Image file not found"}))
        sys.exit(1)

    frame_path = cv2.imread(args.image)
    if frame_path is None:
        print(json.dumps({"error": "Failed to load image"}))
        sys.exit(1)

    # Jalankan prediksi
    result, confidence = predict(frame_path)

    # Hasilkan output JSON
    print(json.dumps({
        "prediction": result,
        "confidence": confidence
    }))