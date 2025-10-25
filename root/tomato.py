from flask import Flask, render_template, request, jsonify
from collections import OrderedDict
import numpy as np
import uuid
import os
from datetime import datetime
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# --- Load Model TFLite ---
interpreter = tf.lite.Interpreter(model_path="./tomato.tflite")
interpreter.allocate_tensors()

# Ambil detail input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Daftar kelas (label) sesuai model ---
class_names = ["bacterial", "early", "mold", "target spot", "yellow", "healthy"]


@app.route('/', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST':
            # Cek apakah file gambar dikirim
            if 'imagefile' not in request.files:
                raise ValueError("No image file found in the request.")

            imagefile = request.files['imagefile']

            if imagefile.filename == '':
                raise ValueError("The uploaded image file is empty.")

            # Baca dan ubah ukuran gambar sesuai input model
            image = Image.open(BytesIO(imagefile.read())).convert('RGB')
            image = image.resize((150, 150))  # Sesuaikan dengan ukuran input model
            image = np.array(image, dtype=np.float32)
            image = image / 255.0  # Normalisasi (pastikan sesuai training model)
            image = np.expand_dims(image, axis=0)

            # Jalankan prediksi TFLite
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            yhat = interpreter.get_tensor(output_details[0]['index'])

            predicted_class = class_names[np.argmax(yhat)]

            # Deskripsi dan tindakan
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

            # Buat response JSON
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

        # Jika GET
        return jsonify({
            "status": "success",
            "message": "Please upload an image for prediction."
        }), 200

    except Exception as e:
        return jsonify({
            "status": "failed",
            "status_code": 400,
            "message": str(e)
        }), 400


@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        "status": "failed",
        "status_code": 404,
        "message": "The requested resource was not found."
    }), 404


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
