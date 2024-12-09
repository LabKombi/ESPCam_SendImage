import cv2
import os
from predictor import predict

# Buka kamera
cap = cv2.VideoCapture(1)

TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()

print("Tekan 'q' untuk keluar.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Simpan frame sebagai file sementara
    temp_image_path = os.path.join(TEMP_DIR, f"frame.jpg")
    cv2.imwrite(temp_image_path, frame)

    # Resize dan konversi warna jika diperlukan
    # resized_frame = cv2.resize(frame, (224, 224))  # Sesuaikan dengan input model

    # Prediksi
    status = predict(temp_image_path)
    print(f"Status: {status}")

    # Tampilkan frame dengan status prediksi
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Prediction", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()