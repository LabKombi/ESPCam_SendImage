import os
import cv2
import mediapipe as mp
import base64
import time
import queue
from threading import Thread
from paho.mqtt import client as mqtt_client
from predictor import predict
from i_proc import draw_bounding_box

# Konfigurasi Adafruit IO
from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

# VideoCaptureBuffer class untuk buffer frame
class VideoCaptureBuffer:
    def __init__(self, src=0, buffer_size=3):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=buffer_size)
        self.stopped = False
        self.thread = Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                self.q.put(frame)
            else:
                self.q.get()  # Drop frame lama

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# Thread untuk prediksi model
class PredictionThread(Thread):
    def __init__(self):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=1)
        self.result = None
        self.stopped = False

    def run(self):
        while not self.stopped:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.result = predict(frame)

    def predict_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def stop(self):
        self.stopped = True
        self.join()

# Fungsi untuk menghubungkan ke broker MQTT
def connect_mqtt():
    client = mqtt_client.Client()
    client.username_pw_set(AIO_USERNAME, AIO_KEY)

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to Adafruit IO")
        else:
            print(f"Failed to connect, return code {rc}")

    client.on_connect = on_connect
    client.connect(AIO_SERVER, 1883)
    return client

# Thread untuk mengirim gambar ke MQTT
class MqttSenderThread(Thread):
    def __init__(self, client, interval=2):
        super().__init__()
        self.client = client
        self.interval = interval
        self.frame_to_send = None
        self.stopped = False

    def run(self):
        while not self.stopped:
            if self.frame_to_send is not None:
                self.send_image(self.frame_to_send)
            time.sleep(self.interval)

    def send_image(self, frame):
        try:
            _, buffer = cv2.imencode(".jpg", frame)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            self.client.publish(AIO_FEED, encoded_image)
        except Exception as e:
            print(f"Error sending image: {e}")

    def update_frame(self, frame):
        self.frame_to_send = frame

    def stop(self):
        self.stopped = True
        self.join()

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Buffer kamera
cap = VideoCaptureBuffer(src=0)
predictor_thread = PredictionThread()
predictor_thread.start()

# MQTT client
mqtt_client = connect_mqtt()
mqtt_sender = MqttSenderThread(mqtt_client)
mqtt_sender.start()

print("Tekan 'q' untuk keluar.")
fps_limit = 10  # FPS limit
prev_time = 0

try:
    while True:
        if cap.q.empty():
            continue  # Tunggu hingga buffer terisi

        frame = cap.read()  # Ambil frame dari buffer
        curr_time = time.time()

        if curr_time - prev_time > 1 / fps_limit:
            prev_time = curr_time

            # Konversi ke RGB untuk MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Deteksi wajah
            results = face_mesh.process(img_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape

                    # Landmark mata kanan
                    right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) \
                                 for i in [33, 160, 158, 133, 153, 144]]
                    draw_bounding_box(frame, right_eye, color=(255, 0, 0))

                    # Landmark mata kiri
                    left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) \
                                for i in [362, 385, 387, 263, 373, 380]]
                    draw_bounding_box(frame, left_eye, color=(255, 0, 0))

                    # Landmark mulut
                    mouth = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) \
                             for i in [78, 191, 80, 13, 312, 308, 324, 318]]
                    draw_bounding_box(frame, mouth, color=(0, 255, 0))

            # Prediksi
            status, confidence_score = predict(frame)
            print(f"Status: {status}")

            # Update frame untuk dikirim ke MQTT
            mqtt_sender.update_frame(frame)

            # Tampilkan status di frame
            if status == "error":
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence_score:.5f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)

            cv2.imshow("Live Prediction", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    # Bersihkan semua resource
    cap.stop()
    predictor_thread.stop()
    mqtt_sender.stop()
    cv2.destroyAllWindows()
