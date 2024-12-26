import os
import cv2  # Untuk membaca gambar sebagai array
import base64
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from paho.mqtt import client as mqtt_client
from predictor import predict

# Konfigurasi Flask
app = Flask(__name__)

CORS(app)

# Konfigurasi Adafruit IO
from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

# Direktori sementara untuk menyimpan file
TEMP_DIR = "temp"

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

# Fungsi untuk membaca dan mengirim gambar ke MQTT
def send_image_to_mqtt(client, image_path):
    try:
        with open(image_path, "rb") as image_file:
            # Encode gambar ke format base64
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Kirim gambar melalui MQTT
        client.publish(AIO_FEED, encoded_image)
        print(f"Image sent to {AIO_FEED}")
    except FileNotFoundError:
        print(f"Error: File {image_path} not found")
    except Exception as e:
        print(f"Error sending image: {e}")

# Endpoint Flask untuk prediksi
@app.route('/predict', methods=['POST'])
def predict_from_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Simpan file yang diterima
    file = request.files['image']
    file_path = os.path.join(TEMP_DIR, file.filename)
    file.save(file_path)

    try:
        # Load gambar dari file menjadi array
        image = cv2.imread(file_path)
        if image is None:
            return jsonify({"error": "Invalid image format or file is corrupted"}), 400

        # Jalankan prediksi
        result, confidence = predict(image)

        # Kirim gambar ke MQTT
        send_image_to_mqtt(mqtt_client_instance, file_path)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        # Hapus file sementara
        if os.path.exists(file_path):
            os.remove(file_path)

    # Kirim hasil prediksi
    return jsonify({"prediction": result, "confidence": confidence})

if __name__ == '__main__':
    # Pastikan direktori sementara ada
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Hubungkan ke MQTT
    mqtt_client_instance = connect_mqtt()
    mqtt_client_instance.loop_start()  # Jalankan loop MQTT di background

    try:
        # Jalankan server Flask
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        mqtt_client_instance.loop_stop()
        mqtt_client_instance.disconnect()
