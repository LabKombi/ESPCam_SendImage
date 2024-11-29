from imports import *


# Import konfigurasi dari file config.py
from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

app = Flask(__name__)

# Adafruit IO Configuration

# Gunakan konfigurasi dalam kode Anda
print("Server:", AIO_SERVER)
print("Username:", AIO_USERNAME)
print("Feed:", AIO_FEED)

# Folder untuk menyimpan foto yang diterima
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model("assets/drowsiness_model.h5")  # grayscale

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EYE_CLOSED_THRESHOLD = 3  
FRAME_COUNT = 0  
last_eye_status = "open" 
start_time = None  

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (64, 64))
    face_normalized = face_resized / 255.0
    face_final = np.expand_dims(face_normalized, axis=-1)
    face_final = np.expand_dims(face_final, axis=0)
    return face_final

# Callback for MQTT connection
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(AIO_FEED)
    else:
        print(f"Failed to connect, return code {rc}")

#  kode untuk penggunaan buffer

# Daftar untuk menyimpan gambar yang diterima
image_buffer = []

# Callback untuk menerima pesan MQTT
def on_message(client, userdata, msg):
    print(f"Received message from MQTT feed: {msg.topic}, Payload size: {len(msg.payload)} bytes")

    try:
        # Jika data dikirim dalam bentuk base64, Anda perlu mendekode
        img_data = base64.b64decode(msg.payload)  # Dekode base64 (jika itu yang digunakan)

        # Mengubah bytes menjadi gambar menggunakan PIL
        image = Image.open(io.BytesIO(img_data))
        
        # Mengubah gambar menjadi format yang bisa diterima oleh OpenCV
        image = np.array(image)

        # Pastikan gambar sudah dalam format yang benar
        if len(image.shape) == 3 and image.shape[2] == 3:  # Memastikan gambar berwarna RGB
            print("Image decoded successfully")
        else:
            print("Invalid image format received")  

    except Exception as e:
        print(f"Error: Failed to decode image from payload. Exception: {str(e)}")
        return

    # Menyimpan gambar ke dalam buffer untuk analisis lebih lanjut
    image_buffer.append(image)

    # Jika sudah ada 3 gambar dalam buffer, lakukan analisis
    if len(image_buffer) >= 3:
        # Analisis status pengemudi berdasarkan 3 gambar
        eye_status_count = {"open": 0, "closed": 0}
        yawn_status_count = {"open": 0, "closed": 0}

        # Proses ketiga gambar
        for img in image_buffer:
            processed_image = preprocess_image(img)
            if processed_image is None:
                print("Error: No face detected in the image.")
                continue

            prediction = model.predict(processed_image)
            eye_status = "closed" if prediction[0][0] > 0.5 else "open"
            print(f"Eye status: {eye_status}")

            # Cek status mulut (yawn detection) jika model mendeteksi mulut terbuka
            if is_yawning(processed_image):  # Anda bisa menambahkan fungsi deteksi mulut terbuka di sini
                yawn_status = "open"  # Mulut terbuka menandakan menguap
            else:
                yawn_status = "closed"  # Mulut tertutup menandakan tidak menguap
            
            # Hitung frekuensi status mata dan mulut
            if eye_status == "open":
                eye_status_count["open"] += 1
            else:
                eye_status_count["closed"] += 1

            if yawn_status == "open":
                yawn_status_count["open"] += 1
            else:
                yawn_status_count["closed"] += 1

        # Tentukan status pengemudi berdasarkan analisis tiga gambar
        if eye_status_count["closed"] == 3:
            print("Pengemudi mengantuk - Mata tertutup di ketiga gambar.")
            send_sleep_status("mengantuk")
        elif yawn_status_count["open"] == 3:
            print("Pengemudi menguap - Mulut terbuka di ketiga gambar.")
            send_sleep_status("menguap")
        elif eye_status_count["closed"] > 0 and yawn_status_count["open"] > 0:
            print("Pengemudi segar - Tidak semua gambar menunjukkan mata tertutup/mulut terbuka.")
            send_sleep_status("segar")
        elif eye_status_count["open"] == 3:
            print("Pengemudi tidak mengantuk - Mata terbuka di ketiga gambar.")
            send_sleep_status("tidak mengantuk")

        # Reset buffer untuk gambar berikutnya
        image_buffer.clear()

# Fungsi untuk mengirim status ke sistem (misalnya menggunakan MQTT atau API)
def send_sleep_status(status):
    print(f"Sending status: {status}")
    # Kode untuk mengirim status (misalnya melalui MQTT atau API)
    # mqtt_client.publish("your_topic", status)

# Fungsi tambahan untuk deteksi mulut terbuka (yawn)
def is_yawning(image):
    # Implementasikan deteksi mulut terbuka pada gambar (gunakan model atau algoritma lainnya)
    # Ini adalah contoh fungsi kosong, Anda dapat menyesuaikannya dengan kebutuhan Anda.
    return False  # Ganti dengan implementasi sebenarnya.

# MQTT Client Setup
mqtt_client = Client()
mqtt_client.username_pw_set(AIO_USERNAME, AIO_KEY)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

mqtt_client.connect(AIO_SERVER, 1883)
mqtt_client.loop_start()

@app.route('/')
def index():
    return "Server is running and connected to MQTT feed."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)