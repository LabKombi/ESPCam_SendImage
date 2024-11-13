# **Alur:**
1. Esp32cam menangkap gambar dan mengirim gambar tersebut menuju server melalui jembatan
2. jembatan menghubungkan esp dengan server
3. server menerima hasil tangkapan gambar melalui jembatan
4. server mengirimkan output berupa perintah relay on kepada firebase
5. firebase menerima perintah
6. perintah pada firebase dibaca oleh esp8266 untuk menghidupkan peringatan
