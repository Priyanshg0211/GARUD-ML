import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'saved_scaler.pkl')

def extract_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def load_data(gunshot_dir, noise_dir):
    features = []
    labels = []
    
    for filename in os.listdir(gunshot_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(gunshot_dir, filename)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(1)  # 1 for gunshot
    
    for filename in os.listdir(noise_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(noise_dir, filename)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(0)  # 0 for noise
    
    return np.array(features), np.array(labels)

if __name__ == '__main__':
    gunshot_dir = os.path.join(BASE_DIR, 'data', 'gunshots')
    noise_dir = os.path.join(BASE_DIR, 'data', 'noise')

    print("Loading and preprocessing data...")
    features, labels = load_data(gunshot_dir, noise_dir)

    if len(features) == 0:
        print("No features extracted. Please check your data directories and file formats.")
    else:
        print(f"Loaded {len(features)} samples.")
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
        print("Training the model...")
        model.fit(X_train_scaled, y_train)

        test_accuracy = model.score(X_test_scaled, y_test)
        print(f"Test accuracy: {test_accuracy}")

        with open(MODEL_PATH, 'wb') as file:
            pickle.dump(model, file)
        with open(SCALER_PATH, 'wb') as file:
            pickle.dump(scaler, file)
        print(f"Model saved to {MODEL_PATH}")
        print(f"Scaler saved to {SCALER_PATH}")