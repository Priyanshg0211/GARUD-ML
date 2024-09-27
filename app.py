import os
from flask import Flask, render_template, request, jsonify, abort
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

app = Flask(__name__)

# Ensure the script works regardless of the current working directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'saved_scaler.pkl')

def extract_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        app.logger.error(f"Error extracting features from {file_path}: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

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
            file.save(temp_path)
            features = extract_features(temp_path)
            if features is None:
                abort(500, description="Error extracting features from the audio file.")
            
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict_proba(features_scaled)[0]
            result = 'Gunshot Detected' if prediction[1] > 0.5 else 'No Gunshot Detected'
            
            return jsonify({'result': result, 'confidence': float(prediction[1])})
        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            abort(500, description="An error occurred during prediction.")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.errorhandler(400)
@app.errorhandler(500)
def handle_error(error):
    return jsonify(error=str(error.description)), error.code

if __name__ == '__main__':
    # Load the model and scaler
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        app.logger.error("Model or scaler file not found. Please train the model first.")
        exit(1)
    except Exception as e:
        app.logger.error(f"Error loading model or scaler: {str(e)}")
        exit(1)
    
    app.run(debug=True)