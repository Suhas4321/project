from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

model_path = os.path.join('C:\\Users\\91897\\OneDrive\\Desktop\\miniproj\\pneumonia_model.h5')
model = load_model(model_path)

# Load the trained model



def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    return prediction[0][0]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        result = predict_pneumonia(file_path)

        if result > 0.5:
            prediction = 'Pneumonia'
        else:
            prediction = 'Normal'

        return render_template('result.html', prediction=prediction)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
