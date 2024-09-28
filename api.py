import os
from flask import Flask, request, jsonify, abort
import numpy as np
import librosa
import pickle

app = Flask(__name__)

# Base directory where the script runs
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Paths to the saved model and scaler
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'saved_scaler.pkl')

# Function to extract MFCC features from the audio file
def extract_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        app.logger.error(f"Error extracting features from {file_path}: {str(e)}")
        return None

# Route to handle the index
@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the Gunshot Detection API'})

# Route to handle audio file prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        abort(400, description="No file part in the request.")
    file = request.files['file']
    if file.filename == '':
        abort(400, description="No selected file.")
    
    if file:
        temp_path = os.path.join(BASE_DIR, 'temp_audio.wav')
        try:
            # Save the uploaded audio file temporarily
            file.save(temp_path)

            # Extract features from the audio file
            features = extract_features(temp_path)
            if features is None:
                abort(500, description="Error extracting features from the audio file.")
            
            # Scale the features before making the prediction
            features_scaled = scaler.transform(features.reshape(1, -1))
            
            # Get prediction probabilities from the model
            prediction = model.predict_proba(features_scaled)[0]
            result = 'Gunshot Detected' if prediction[1] > 0.5 else 'No Gunshot Detected'
            
            # Return the prediction result and confidence
            return jsonify({
                'result': result, 
                'confidence': float(prediction[1])
            })
        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            abort(500, description="An error occurred during prediction.")
        finally:
            # Remove the temporary file after prediction
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Custom error handler for better error messages
@app.errorhandler(400)
@app.errorhandler(500)
def handle_error(error):
    return jsonify(error=str(error.description)), error.code

if __name__ == '__main__':
    # Load the trained model and scaler
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
        app.logger.info("Model and Scaler loaded successfully.")
    except FileNotFoundError:
        app.logger.error("Model or scaler file not found. Please ensure they are present in the correct path.")
        exit(1)
    except Exception as e:
        app.logger.error(f"Error loading model or scaler: {str(e)}")
        exit(1)
    
    # Start the Flask app
    app.run(debug=True)
