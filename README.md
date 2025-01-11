Early Detection of Pneumonia and Tuberculosis Using Chest X-Ray Images

Overview
This project is a web-based application designed to assist in the early detection of Pneumonia and Tuberculosis (TB) using Chest X-Ray images. Powered by deep learning models and an intuitive user interface, the tool offers fast, reliable, and accurate diagnoses to help medical professionals and researchers.

Features
Disease Selection: Diagnose for Pneumonia, Tuberculosis, or both simultaneously.
Image Upload: Upload a Chest X-Ray image to perform the diagnosis.

Deep Learning Models:
A specialized Convolutional Neural Network (CNN) trained for Pneumonia detection.
Another CNN trained for Tuberculosis detection.

Real-Time Feedback:
Displays results with confidence scores.
Option to preview the uploaded image.

Clear Functionality: Reset the form to upload a new X-ray or change the disease selection.

Responsive Design: User-friendly interface built with Flask and Bootstrap.

Datasets Used

1.Pneumonia Detection Dataset:

Source: Kaggle (RSNA Pneumonia Detection Challenge).
Size: Over 5,800 X-Ray images split into training, validation, and testing sets.

2.Tuberculosis Detection Dataset:

Source: Kaggle - Tuberculosis Chest X-Ray Dataset
Size: Contains over 3,500 X-Ray images split into training, validation, and testing sets.

Model Architecture
1.Pneumonia Model:

A CNN with multiple convolutional, pooling, and dense layers.
Input size: 180x180 grayscale images.
Achieved high validation accuracy during training.

2.Tuberculosis Model:

A CNN designed for multi-class classification.
Input size: 180x180 RGB images.
Evaluated for precision and recall to handle real-world TB cases.

Technologies Used

Frontend:
HTML, CSS, Bootstrap 5
AJAX and jQuery for real-time requests and modal displays.

Backend:
Flask (Python framework)
TensorFlow/Keras for deep learning models.

Other Libraries:
PIL for image preprocessing.
NumPy for numerical computations.

How It Works
The user uploads a Chest X-Ray image through the web interface.
Selects the disease type: Pneumonia, Tuberculosis, or Both.
The image is preprocessed and fed into the appropriate deep learning model(s).
The model predicts the likelihood of the selected disease(s) and displays the results along with confidence scores.

How to Run the Project
1.Clone the repository:
git clone https://github.com/your-username/early-detection-xray.git
cd early-detection-xray

2.Install dependencies:
pip install -r requirements.txt

3.Set up the models:

Place the Pneumonia and TB models in the Models/ directory.
Ensure the files are named pneumonia_detection_model.keras and TB_CNN_Model.keras.

4.Start the Flask application:
python app.py

5.Open the application in your browser:
http://127.0.0.1:5000

Results

Pneumonia Model:

Validation Accuracy: ~90%
Trained on 5,800+ chest X-rays.

Tuberculosis Model:

Validation Accuracy: ~85%
Trained on 3,500+ chest X-rays.

Future Enhancements
Add support for additional diseases.
Implement Grad-CAM visualization for model interpretability.
Deploy the application on cloud platforms like Heroku or AWS for global accessibility.
Improve accuracy by using ensemble models.

Contributors
R.D.V.Pahasarani
Role: Developer
Contact: 
email - vindulapahasarani@gmail.com

This project is an essential tool for researchers and medical practitioners in the fight against Pneumonia and Tuberculosis. If you find it helpful, please ⭐ the repository! 😊
