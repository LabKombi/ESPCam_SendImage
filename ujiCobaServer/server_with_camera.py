import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from imutils import face_utils
import dlib

# Load model dan detektor wajah/landmark
model = load_model("model/mobilenet_rgb.h5")
landmark_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

CLASSES = ["not_sleepy", "sleepy", "yawn"]

EYE_CLOSED_THRESHOLD = 1.5
YAWN_THRESHOLD = 3.0

eye_closed_start_time = None
yawn_start_time = None

def preprocess_frame(frame, target_size=(64, 64), feature="eye"):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces) == 0:
        return None, None

    for face in faces:
        landmarks = landmark_predictor(gray, face)
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
            return None, None

        margin = 15
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(gray.shape[1], x_max + margin)
        y_max = min(gray.shape[0], y_max + margin)

        # region = gray[y_min:y_max, x_min:x_max]
        # region_resized = cv2.resize(region, target_size, interpolation=cv2.INTER_CUBIC)
        # region_normalized = region_resized / 255.0
        # return np.expand_dims(region_normalized, axis=(0, -1)), face

        # Convert region from grayscale to RGB
        region = gray[y_min:y_max, x_min:x_max]
        region_rgb = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)  # Convert to RGB

        region_resized = cv2.resize(region_rgb, target_size, interpolation=cv2.INTER_CUBIC)
        region_normalized = region_resized / 255.0
        return np.expand_dims(region_normalized, axis=(0, -1)), face

    return None, None

def main():
    global eye_closed_start_time, yawn_start_time

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = "not_sleepy"

        # Preprocess untuk mata
        preprocessed_eye_image, face = preprocess_frame(frame, feature="eye")
        if preprocessed_eye_image is not None:
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

        # Preprocess untuk mulut
        preprocessed_mouth_image, _ = preprocess_frame(frame, feature="mouth")
        if preprocessed_mouth_image is not None:
            predictions = model.predict(preprocessed_mouth_image)
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

        # Tampilkan hasil prediksi
        if face is not None:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Prediction: {result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Sleepiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
