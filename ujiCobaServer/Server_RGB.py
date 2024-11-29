from imports import *
from config.config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

app = Flask(__name__)

# Folder to save photos
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model("drowsiness_detection_model_after.h5")  # RGB

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    # Ubah gambar grayscale menjadi gambar RGB dengan menduplikasi saluran
    face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

    # Resize dan normalisasi gambar
    face_resized = cv2.resize(face_rgb, (64, 64))
    face_normalized = face_resized / 255.0

    # Pastikan input memiliki dimensi yang benar untuk model (1, 64, 64, 3)
    face_final = np.expand_dims(face_normalized, axis=0)  # Menambah dimensi batch
    return face_final


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(AIO_FEED)
    else:
        print(f"Failed to connect, return code {rc}")


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

    # Save the image locally for debugging
    cv2.imwrite(os.path.join(UPLOAD_FOLDER, "mqtt_image.jpg"), image)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    if processed_image is None:
        print("Error: No face detected in the image.")
        return

    prediction = model.predict(processed_image)
    eye_status = "closed" if prediction[0][0] > 0.5 else "open"
    print(f"Eye status: {eye_status}")

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
