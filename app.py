import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
predictions = [0, 0]

# Define the paths for the models
pneumonia_model_path = r"Models/pneumonia_detection_model.keras"
tb_model_path = r"Models/TB_CNN_Model.keras"

# Check if model files exist
if not os.path.exists(pneumonia_model_path):
    raise FileNotFoundError(f"Pneumonia model file not found at: {pneumonia_model_path}")

if not os.path.exists(tb_model_path):
    raise FileNotFoundError(f"TB model file not found at: {tb_model_path}")

# Load both models
pneumonia_model = load_model(pneumonia_model_path)
tb_model = load_model(tb_model_path)

def preprocess_image_pneumonia(img):
    img = img.resize((180, 180)).convert('L')
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = img_array / 255.0
    img_array = img_array.reshape(-1, 180, 180, 1)
    return img_array

def preprocess_image_tb(img):
    img = img.resize((180, 180)).convert('RGB')
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_pneumonia(img):
    img_array = preprocess_image_pneumonia(img)
    prediction = pneumonia_model.predict(img_array)[0][0]
    predictions[0] = prediction
    confidence = round((1 - prediction) * 100, 2)
    if prediction < 0.5:
        return 'Pneumonia detected', confidence
    else:
        return 'No pneumonia detected', confidence

def predict_tb(img):
    img_array = preprocess_image_tb(img)
    prediction = tb_model.predict(img_array)[0][0]
    predictions[1] = prediction
    confidence = round(prediction * 100, 2)
    if prediction > 0.5:
        return 'Tuberculosis detected', confidence
    else:
        return 'No tuberculosis detected', confidence

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        img = Image.open(io.BytesIO(file.read()))
        selected_disease = request.form['disease']
        
        # Save the file temporarily to display it later
        temp_file_path = os.path.join('uploads', file.filename)
        img.save(temp_file_path)

        if selected_disease == "Pneumonia":
            result, confidence = predict_pneumonia(img)
        elif selected_disease == "Tuberculosis":
            result, confidence = predict_tb(img)
        elif selected_disease == "Both":
            result_pneumonia, confidence_pneumonia = predict_pneumonia(img)
            result_tb, confidence_tb = predict_tb(img)
            result = f"{result_pneumonia} ({confidence_pneumonia}%), {result_tb} ({confidence_tb}%)"
            confidence = max(confidence_pneumonia, confidence_tb)

        return jsonify({
            'result': result,
            'confidence': confidence,
            'image_file': file.filename  # Return the filename for displaying the image
        })

if __name__ == '__main__':
    app.run(debug=True)