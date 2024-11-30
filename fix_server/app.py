import os
from flask import Flask, request, jsonify
from predictor import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_from_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    try:
        # Jalankan prediksi dengan path file gambar
        result = predict(file_path)
    except Exception as e:
        # Tangani error saat proses prediksi
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally: # dapat dihilangkan jika tidak ingin menghapus file sementara
        # Hapus file sementara setelah selesai
        if os.path.exists(file_path):
            os.remove(file_path)

    return jsonify({"prediction": result})

if __name__ == '__main__':
    if not os.path.exists("temp"):
        os.makedirs("temp")
    app.run(debug=True)
