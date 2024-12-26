import cv2
import requests

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
        with open(temp_image_path, 'rb') as img_file:
            files = {'image': img_file}
            response = requests.post(API_URL, files=files)

        # Tampilkan prediksi dari server
        if response.status_code == 200:
            try:
                prediction = response.json()
                if all(key in prediction for key in ["prediction", "eye_confidence", "mouth_confidence"]):
                    print(f"Prediction: {prediction['prediction']}, "
                          f"Eye Confidence: {prediction['eye_confidence']}, "
                          f"Mouth Confidence: {prediction['mouth_confidence']}")

                    # Tampilkan informasi di frame
                    cv2.putText(frame, f"Prediction: {prediction['prediction']}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Eye left Conf: {prediction['eye_confidence']:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    print("Invalid response format:", prediction)

            except Exception as e:
                print("Error parsing response:", e)
        else:
            print(f"Error: {response.json().get('error', 'Unknown error')}")

        # Tampilkan frame
        cv2.imshow("Camera Feed", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan kamera dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_predict()

    app.run(debug=True)
