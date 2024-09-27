import tensorflow as tf
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
import sys

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

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
    
    for directory, label in [(gunshot_dir, 1), (noise_dir, 0)]:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):
                file_path = os.path.join(directory, filename)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(label)
    
    return np.array(features), np.array(labels)

def build_model(input_shape):
    """Build a CNN model for gunshot vs noise classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main execution
if __name__ == "__main__":
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gunshot_dir = os.path.join(script_dir, 'data', 'gunshots')
    noise_dir = os.path.join(script_dir, 'data', 'noise')

    print("Loading and preprocessing data...")
    features, labels = load_data(gunshot_dir, noise_dir)

    if len(features) == 0:
        print("No features extracted. Please check your data directories and file formats.")
        sys.exit(1)

    print(f"Loaded {len(features)} samples.")
    
    # Normalize features
    features = (features - np.mean(features)) / np.std(features)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Build and compile the model
    input_shape = (X_train.shape[1],)
    model = build_model(input_shape)
    model.summary()

    # Train the model
    print("Training the model...")
    
    # Convert numpy arrays to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_size = int(0.2 * len(X_train))
    val_dataset = train_dataset.take(val_size)
    train_dataset = train_dataset.skip(val_size)

    # Set steps per epoch and validation steps
    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy() or 1
    validation_steps = tf.data.experimental.cardinality(val_dataset).numpy() or 1

    try:
        history = model.fit(train_dataset, epochs=20, validation_data=val_dataset, 
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    except Exception as e:
        print(f"Error during training: {e}")

    # Evaluate the model
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model to TensorFlow format with .keras extension
    model_save_path = os.path.join(script_dir, 'model', 'gunshot_classifier.keras')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Load the model from TensorFlow format
    try:
        loaded_model = tf.keras.models.load_model(model_save_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

    # Example usage of the loaded model for prediction
    if len(X_test) > 0:
        sample_feature = X_test[0].reshape(1, -1)  # Reshape for a single prediction
        prediction = loaded_model.predict(sample_feature)
        print(f"Predicted probability of gunshot: {prediction[0][0]:.4f}")
