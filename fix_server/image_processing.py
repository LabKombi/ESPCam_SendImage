import os
import cv2
import numpy as np
import dlib
from imutils import face_utils

# Folder untuk menyimpan foto yang diterima
FOLDER_0 = './debug'
FOLDER_1 = './debug/eye_r'
FOLDER_2 = './debug/eye_l'
FOLDER_3 = './debug/mouth'

landmark_predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

for folder in [FOLDER_1, FOLDER_2, FOLDER_3]:
    os.makedirs(folder, exist_ok=True)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def remove_noise_bilateral(image, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def normalize_image(image):
    return image / 255.0

def preprocess_image(image_path, target_size=(224, 224), feature="eye_r"):
    img_1 = cv2.imread(image_path)
    if img_1 is None:
        raise ValueError("Image not found or unreadable")

    img_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)

    cv2.imwrite(os.path.join(FOLDER_0,"debug_asli.jpg"), img_1)
    cv2.imwrite(os.path.join(FOLDER_0,"debug_rgb.jpg"), img_2)

    # extract right eye region
    if feature == "eye_r":

        img_rgb = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        
        detector = dlib.get_frontal_face_detector()
        faces = detector(img_rgb, 1)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        for face in faces:
            landmarks = landmark_predictor(img_rgb, face)
            landmarks = face_utils.shape_to_np(landmarks)
            eye_points = landmarks[36:42]
            x_min, y_min = eye_points.min(axis=0)
            x_max, y_max = eye_points.max(axis=0)

            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img_rgb.shape[1], x_max + margin)
            y_max = min(img_rgb.shape[0], y_max + margin)

            cropped_img = img_rgb[y_min:y_max, x_min:x_max]
            denoised_img = remove_noise_bilateral(cropped_img)
            sharpened_img = apply_sharpening(denoised_img)
            resized_img = cv2.resize(sharpened_img, target_size, interpolation=cv2.INTER_CUBIC)
            normalized_img = normalize_image(resized_img)

            # Debugging: Save intermediate results
            cv2.imwrite(os.path.join(FOLDER_1,"debug_eye_r_rgb.jpg"), img_rgb)
            cv2.imwrite(os.path.join(FOLDER_1,"debug_eye_r_cropped.jpg"), cropped_img)
            cv2.imwrite(os.path.join(FOLDER_1,"debug_eye_r_denoised.jpg"), denoised_img)
            cv2.imwrite(os.path.join(FOLDER_1,"debug_eye_r_sharpened.jpg"), sharpened_img)
            cv2.imwrite(os.path.join(FOLDER_1,"debug_eye_r_resized.jpg"), resized_img)

        return np.expand_dims(normalized_img, axis=0)

    if feature == "eye_l":
        img_rgb = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

        detector = dlib.get_frontal_face_detector()
        faces = detector(img_rgb, 1)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        for face in faces:
            landmarks = landmark_predictor(img_rgb, face)
            landmarks = face_utils.shape_to_np(landmarks)
            eye_points = landmarks[42:48]
            x_min, y_min = eye_points.min(axis=0)
            x_max, y_max = eye_points.max(axis=0)

            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img_rgb.shape[1], x_max + margin)
            y_max = min(img_rgb.shape[0], y_max + margin)

            cropped_img = img_rgb[y_min:y_max, x_min:x_max]
            denoised_img = remove_noise_bilateral(cropped_img)
            sharpened_img = apply_sharpening(denoised_img)
            resized_img = cv2.resize(sharpened_img, target_size, interpolation=cv2.INTER_CUBIC)
            normalized_img = normalize_image(resized_img)

            # Debugging: Save intermediate results
            cv2.imwrite(os.path.join(FOLDER_2,"debug_eye_l_rgb.jpg"), img_rgb)
            cv2.imwrite(os.path.join(FOLDER_2,"debug_eye_l_cropped.jpg"), cropped_img)
            cv2.imwrite(os.path.join(FOLDER_2,"debug_eye_l_denoised.jpg"), denoised_img)
            cv2.imwrite(os.path.join(FOLDER_2,"debug_eye_l_sharpened.jpg"), sharpened_img)
            cv2.imwrite(os.path.join(FOLDER_2,"debug_eye_l_resized.jpg"), resized_img)

        return np.expand_dims(normalized_img, axis=0)

    elif feature == "mouth":
        img_rgb = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)

        detector = dlib.get_frontal_face_detector()
        faces = detector(img_rgb, 1)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")

        for face in faces:
            landmarks = landmark_predictor(img_rgb, face)
            landmarks = face_utils.shape_to_np(landmarks)
            mouth_points = landmarks[48:68]
            x_min, y_min = mouth_points.min(axis=0)
            x_max, y_max = mouth_points.max(axis=0)

            margin = 15
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img_rgb.shape[1], x_max + margin)
            y_max = min(img_rgb.shape[0], y_max + margin)

            cropped_img = img_rgb[y_min:y_max, x_min:x_max]
            denoised_img = remove_noise_bilateral(cropped_img)
            sharpened_img = apply_sharpening(denoised_img)
            resized_img = cv2.resize(sharpened_img, target_size, interpolation=cv2.INTER_CUBIC)
            normalized_img = normalize_image(resized_img)

            # Debugging: Save intermediate results
            cv2.imwrite(os.path.join(FOLDER_3,"debug_mouth_rgb.jpg"), img_rgb)
            cv2.imwrite(os.path.join(FOLDER_3,"debug_mouth_cropped.jpg"), cropped_img)
            cv2.imwrite(os.path.join(FOLDER_3,"debug_mouth_denoised.jpg"), denoised_img)
            cv2.imwrite(os.path.join(FOLDER_3,"debug_mouth_sharpened.jpg"), sharpened_img)
            cv2.imwrite(os.path.join(FOLDER_3,"debug_mouth_resized.jpg"), resized_img)

            return np.expand_dims(normalized_img, axis=0)

    else:
        raise ValueError("Invalid feature type")
