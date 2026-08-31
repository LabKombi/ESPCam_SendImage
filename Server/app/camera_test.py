import cv2
from predictor import predict
import time

# cv2.CAP_DSHOW if using Windows, it will be more stable.

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak ditemukan!")
    exit()

print("Kamera aktif! Tekan 'q' untuk keluar.")

# --- PENGATURAN FRAME SKIPPING ---
frame_count = 0
process_every_n_frames = 10  # The predictor will only "think" once every 5 frames.
                             # If still hard, change to 10. if run properly, change to 2 or 3.
last_text = "Menganalisis..."
last_color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil gambar dari kamera.")
        break

    frame_count += 1

    # Predictor only run on conditioned frame (so CPU not wrking hard)
    if frame_count % process_every_n_frames == 0:
        try:
            result, confidence = predict(frame)
            
            if result == "error":
                last_text = "No Face Detected"
                last_color = (0, 165, 255) # Orange
            else:
                # Number format for confidence 2 decimal only (example: 0.91)
                last_text = f"{result} ({confidence:.2f})"
                last_color = (0, 255, 0) if "not_sleepy" in result else (0, 0, 255)
                
        except Exception as e:
            last_text = "Processing Error"
            last_color = (0, 0, 255)

    # Paste text (using the latest detection result)
    cv2.putText(frame, last_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, last_color, 2, cv2.LINE_AA)

    # Show the video frame with the detection result
    cv2.imshow("Testing Model Deteksi Kantuk", frame)

    # Read keyboard input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()