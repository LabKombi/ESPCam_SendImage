import os
import cv2
import numpy as np
import mediapipe as mp

# Folder untuk menyimpan foto yang diterima
FOLDER_0 = './debug'
FOLDER_1 = './debug/eye_r'
FOLDER_2 = './debug/eye_l'
FOLDER_3 = './debug/mouth'

for folder in [FOLDER_0, FOLDER_1, FOLDER_2, FOLDER_3]:
    os.makedirs(folder, exist_ok=True)

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Fungsi tambahan
def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def remove_noise_bilateral(image, diameter=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def normalize_image(image):
    return image / 255.0

def preprocess_image(frame, target_size=(224, 224), feature="eye_r"):
    img = frame.copy()
    cv2.imwrite(os.path.join(FOLDER_0, "debug_asli_1.jpg"), img)

    if img is None:
        raise ValueError("Image not found or unreadable")

    # Konversi ke RGB untuk MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Deteksi wajah menggunakan MediaPipe
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        raise ValueError("No face detected in the image")

    # Ambil landmark wajah pertama (karena hanya satu wajah)
    landmarks = results.multi_face_landmarks[0]

    # Ambil indeks landmark sesuai fitur
    if feature == "eye_r":
        indices = [33, 160, 158, 133, 153, 144]  # Landmark mata kanan
    elif feature == "eye_l":
        indices = [362, 385, 387, 263, 373, 380]  # Landmark mata kiri
    elif feature == "mouth":
        indices = [78, 191, 80, 13, 312, 308, 324, 318]  # Landmark mulut
    else:
        raise ValueError("Invalid feature type")

    # Ambil koordinat landmark
    h, w, _ = img.shape
    points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    margin = 15
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    cropped_img = img[y_min:y_max, x_min:x_max]
    denoised_img = remove_noise_bilateral(cropped_img)
    sharpened_img = apply_sharpening(denoised_img)
    resized_img = cv2.resize(sharpened_img, target_size, interpolation=cv2.INTER_CUBIC)
    normalized_img = normalize_image(resized_img)

    return np.expand_dims(normalized_img, axis=0)

def draw_bounding_box(image, points, color=(0, 255, 0), thickness=2):
    points = np.array(points)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    margin = 15
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image
