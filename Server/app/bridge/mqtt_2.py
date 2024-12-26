import requests
import base64
from io import BytesIO
from PIL import Image
import time

from config_aio import AIO_SERVER, AIO_USERNAME, AIO_KEY, AIO_FEED

# URL API Adafruit IO untuk mengambil feed
url = "https://io.adafruit.com/api/v2/{AIO_USERNAME}/feeds?x-aio-key={AIO_KEY}"

# URL server lokal untuk mengirim gambar
server_url = "http://127.0.0.1:5000/predict"

while True:
    try:
        # Mengambil data dari API Adafruit IO
        response = requests.get(url)

        if response.status_code == 200:
            # Parse data JSON
            feeds = response.json()

            # Cari feed dengan ID
            feed_data = None
            for feed in feeds:
                if feed['id'] == 2940873:
                    feed_data = feed
                    break

            if feed_data:
                # Ambil last_value yang berisi gambar dalam format base64
                base64_image_data = feed_data.get('last_value')

                if base64_image_data:
                    # Decode gambar dari base64
                    image_data = base64.b64decode(base64_image_data)

                    # Mengonversi byte image ke format gambar
                    image = Image.open(BytesIO(image_data))

                    # Mengirim gambar ke server lokal
                    files = {'image': ('photo.jpg', image_data, 'image/jpeg')}
                    res = requests.post(server_url, files=files)

                    # Menampilkan hasil dari server
                    print(res.json())
                else:
                    print("Gambar tidak ditemukan dalam 'last_value'.")
            else:
                print("Feed dengan ID 2940873 tidak ditemukan.")
        else:
            print(f"Gagal mengambil data dari API. Status kode: {response.status_code}")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

    # Tunggu 1 detik sebelum mengulangi proses
    time.sleep(0.5)
