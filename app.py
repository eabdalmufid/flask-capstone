import os
from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Labels untuk prediksi
Labels = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis',
          'Dermatofibroma', 'Melanocytic nevus', 'Melanoma',
          'Squamous cell carcinoma', 'Vascular lesions']

confidence_threshold = 60
# Memuat model
model = load_model('./hasilkeduabersih_terbaik.keras')

# Inisialisasi Flask
app = Flask(__name__)

# Fungsi untuk memproses gambar dan prediksi
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = Labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100 

    if confidence < confidence_threshold:
        predicted_class = "Tidak terindikasi"
        confidence = 0

    return predicted_class, confidence, img

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Cek apakah ada parameter query 'result'
    result_type = request.args.get('result', 'text')  # Default ke 'text' jika tidak ada
    
    # Memeriksa apakah ada file gambar yang diupload
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Membaca file gambar
    img = Image.open(file.stream)
    
    # Menyimpan file sementara untuk proses prediksi
    img_path = 'uploaded.jpg'
    img.save(img_path)
    
    # Melakukan prediksi
    predicted_class, confidence, img = predict_image(img_path)

    # Membuat kode label (huruf kecil dan spasi diganti dengan underscore)
    label_code = predicted_class.lower().replace(" ", "_")

    # Hasil dalam format text
    result = {
        'predicted_class': predicted_class,
        'confidence': float(confidence),  # Ubah ke tipe float biasa
        'code': label_code  # Kode label dengan huruf kecil dan underscore
    }

    # Jika result_type adalah 'image', kirim gambar
    if result_type == 'image':
        # Menampilkan gambar dengan judul dan prediksi menggunakan matplotlib
        plt.imshow(img)
        plt.title(f"{predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        img_byte_arr = io.BytesIO()
        plt.savefig(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        plt.close()

        # Mengembalikan gambar sebagai response
        return send_file(img_byte_arr, mimetype='image/jpeg')

    # Jika result_type adalah 'text' (default), kembalikan hasil JSON
    return jsonify(result)


# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)