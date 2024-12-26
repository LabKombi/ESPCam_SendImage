import os
import cv2
import mediapipe as mp
import base64
import time
import queue
from threading import Thread
from paho.mqtt import client as mqtt_client
from i_proc import draw_bounding_box

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
cap = VideoCaptureBuffer(src=1)

# MQTT client
mqtt_client = connect_mqtt()
mqtt_sender = MqttSenderThread(mqtt_client)
mqtt_sender.start()

print("Tekan 'q' untuk keluar.")
fps_limit = 15  # FPS limit
prev_time = 0

try:
    while True:
        if cap.q.empty():
            continue  # Tunggu hingga buffer terisi

        frame = cap.read()  # Ambil frame dari buffer
        curr_time = time.time()

        if curr_time - prev_time > 2 / fps_limit:
            prev_time = curr_time

            # Konversi ke RGB untuk MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Deteksi wajah menggunakan MediaPipe
            results = face_mesh.process(img_rgb)
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape

                    # Ambil koordinat mata kanan
                    eye_r_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in [33, 160, 158, 133, 153, 144]]
                    draw_bounding_box(frame, eye_r_points, color=(255, 0, 0))

                    # Ambil koordinat mata kiri
                    eye_l_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in [362, 385, 387, 263, 373, 380]]
                    draw_bounding_box(frame, eye_l_points, color=(255, 0, 0))

                    # Ambil koordinat mulut
                    mouth_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in [78, 191, 80, 13, 312, 308, 324, 318]]
                    draw_bounding_box(frame, mouth_points, color=(0, 255, 0))

            # Update frame untuk dikirim ke MQTT
            mqtt_sender.update_frame(frame)

            # Tampilkan frame
            cv2.imshow("Live Detection", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    # Bersihkan semua resource
    cap.stop()
    mqtt_sender.stop()
    cv2.destroyAllWindows()
