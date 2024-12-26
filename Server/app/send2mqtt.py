import time
import base64
from paho.mqtt import client as mqtt_client

from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

# Konfigurasi gambar yang akan dikirim
IMAGE_PATH = "debug/debug_asli_1.jpg"  # Ganti dengan path gambar yang akan dikirim
SEND_INTERVAL = 2  # Interval pengiriman dalam detik

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

# Fungsi untuk membaca dan mengirim gambar sebagai string base64
def send_image(client):
    try:
        with open(IMAGE_PATH, "rb") as image_file:
            # Encode gambar ke format base64
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Kirim gambar melalui MQTT
        client.publish(AIO_FEED, encoded_image)
        print(f"Image sent to {AIO_FEED}")
    except FileNotFoundError:
        print(f"Error: File {IMAGE_PATH} not found")
    except Exception as e:
        print(f"Error sending image: {e}")

# Main loop untuk pengiriman gambar secara periodik
def main():
    client = connect_mqtt()
    client.loop_start()  # Mulai loop MQTT

    try:
        while True:
            send_image(client)
            time.sleep(SEND_INTERVAL)
    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
