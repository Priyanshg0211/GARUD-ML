import tensorflow as tf
from tensorflow.python.keras import layers, models
import numpy as np
import librosa
import os

def extract_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """Extract MFCC features from an audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def load_data(gunshot_dir, noise_dir):
    """Load and preprocess the dataset."""
    features = []
    labels = []
    
    # Process gunshot files
    for filename in os.listdir(gunshot_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(gunshot_dir, filename)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(1)  # 1 for gunshot
    
    # Process noise files
    for filename in os.listdir(noise_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(noise_dir, filename)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(0)  # 0 for noise
    
    return np.array(features), np.array(labels)

def build_model(input_shape):
    """Build a CNN model for gunshot vs noise classification."""
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess the data
script_dir = os.path.dirname(os.path.abspath(__file__))
gunshot_dir = os.path.join(script_dir, 'data', 'gunshots')
noise_dir = os.path.join(script_dir, 'data', 'noise')

print("Loading and preprocessing data...")
features, labels = load_data(gunshot_dir, noise_dir)

if len(features) == 0:
    print("No features extracted. Please check your data directories and file formats.")
else:
    print(f"Loaded {len(features)} samples.")
    
    # Normalize features
    features = (features - np.mean(features)) / np.std(features)

    # Build and compile the model
    input_shape = (features.shape[1],)
    model = build_model(input_shape)
    model.summary()

    # Train the model
    print("Training the model...")
    history = model.fit(features, labels, validation_split=0.2, epochs=50, batch_size=32)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(features, labels)
    print(f"Test accuracy: {test_accuracy}")

    # model.py
import pickle

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
    
    def load_model(self):
        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    
    def predict(self, data):
        # Ensure your data is preprocessed and fed into the model correctly
        prediction = self.model.predict(data)
        return prediction
