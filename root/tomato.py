from flask import Flask, render_template, request, jsonify
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from collections import OrderedDict
import numpy as np
import uuid
import os
from datetime import datetime
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Memuat model buatan sendiri
model = load_model('./tomato.h5')



@app.route('/', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            # Memastikan file gambar ada di request
            if 'imagefile' not in request.files:
                raise ValueError("No image file found in the request.")

            imagefile = request.files['imagefile']
            print(imagefile)

            # Validasi file tidak kosong
            if imagefile.filename == '':
                raise ValueError("The uploaded image file is empty.")

            # image_path = "./images/" + imagefile.filename
            # imagefile.save(image_path)

            # Memuat dan mengubah ukuran gambar sesuai dengan input model
            image = Image.open(BytesIO(imagefile.read()))
            image = image.resize((150,150))  # Sesuaikan dengan ukuran input model Anda
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))  # Menyesuaikan dimensi input

            # Melakukan prediksi
            yhat = model.predict(image)

            # Mengasumsikan output model berupa probabilitas untuk beberapa kelas
            class_names = ["bacterial", "early", "mold", "target spot", "yellow", "healthy"]  # Pastikan sesuai dengan label model
            predicted_class = class_names[np.argmax(yhat)]  # Memilih kelas dengan probabilitas tertinggi

            
            # Description and actions based on class
            if predicted_class == "healthy":
                description = "Green, medium to large leaves."
                action = "Water 1-2 times daily, add compost biweekly, ensure 4-6 hours of sunlight."
            elif predicted_class == "yellow":
                description = "Yellowing leaves, often starting from the bottom."
                action = "Add NPK fertilizer, reduce watering, remove yellow leaves."
            elif predicted_class == "target spot":
                description = "Yellow or brown spots on leaves, expanding and merging."
                action = "Prune infected leaves, use fungicides, ensure proper spacing."
            elif predicted_class == "mold":
                description = "Brown, black, or gray spots with white fungal layer."
                action = "Clean the garden, space plants well, use fungicides, remove infected leaves."
            elif predicted_class == "early":
                description = "Brown spots with dark concentric rings."
                action = "Prune infected leaves, use fungicides, ensure good air circulation."
            elif predicted_class == "bacterial":
                description = "Wilting leaves, brown/black spots, rotten fruit."
                action = "Remove infected plants, use antibiotics (with guidance), ensure proper spacing."
            else:
                description = "Unknown class"
                action = "No action available"


            # Menyusun response dalam format JSON
            response = OrderedDict({
                "status": "success",
                "status_code": 200,
                "id": str(uuid.uuid4()),
                "createdAt": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "label": predicted_class,
                "description": description,
                "action": action,
                "message": "The image was successfully processed and classified."
            })

            return jsonify(response)

        return jsonify({
            "status": "success",
            "message": "Please upload an image for prediction."
        }), 200

    except Exception as e:
        # Menyusun response untuk error
        response = {
            "status": "failed",
            "status_code": 400,  # Bad Request
            "message": str(e)
        }
        return jsonify(response), 400

# Penanganan error global untuk route yang tidak ditemukan
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        "status": "failed",
        "status_code": 404,
        "message": "The requested resource was not found."
    }), 404

# Penanganan error global untuk error server
@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "status": "failed",
        "status_code": 500,
        "message": "An internal server error occurred. Please try again later."
    }), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
