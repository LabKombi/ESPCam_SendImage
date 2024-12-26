import cv2
import requests
import os

# Alamat endpoint Flask
API_URL = "http://127.0.0.1:5000/predict"

def capture_and_predict():
    cap = cv2.VideoCapture(0)  # 0 adalah default kamera
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera")
        return

    print("Tekan 'q' untuk keluar")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera")
            break

        # Simpan frame sementara
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Kirim frame ke server Flask
        try:
            with open(temp_image_path, 'rb') as img_file:
                files = {'image': img_file}
                response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                prediction = response.json()
                if "prediction" in prediction and "confidence" in prediction:
                    pred_result = prediction['prediction']
                    confidence = prediction['confidence']
                    print(f"Prediction: {pred_result}, Confidence: {confidence:.2f}")

                    # Tampilkan informasi di frame
                    cv2.putText(frame, f"Prediction: {pred_result}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    print("Invalid response format:", prediction)
            else:
                print(f"Error: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error during request: {e}")

        # Tampilkan frame
        cv2.imshow("Camera Feed", frame)

        # Hapus file sementara
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan kamera dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_predict()
