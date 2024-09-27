import os
import numpy as np
import librosa
import soundfile as sf

def preprocess_audio(audio_dir, output_dir, sample_rate=16000, duration=2):
    """Preprocess and save audio files for ML model."""
    if not os.path.exists(audio_dir):
        print(f"Error: The directory {audio_dir} does not exist.")
        return

    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    files_processed = 0
    files_skipped = 0

    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            try:
                # Load and trim/pad audio to a fixed duration
                filepath = os.path.join(audio_dir, filename)
                data, sr = librosa.load(filepath, sr=sample_rate)
                max_samples = int(sample_rate * duration)
                if len(data) > max_samples:
                    data = data[:max_samples]
                else:
                    data = np.pad(data, (0, max_samples - len(data)))
                
                # Save processed audio
                output_path = os.path.join(output_dir, filename)
                sf.write(output_path, data, sample_rate)
                files_processed += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                files_skipped += 1
        else:
            files_skipped += 1

    print(f"Processed {files_processed} files.")
    print(f"Skipped {files_skipped} files.")

# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))
gunshot_dir = os.path.join(script_dir, 'data', 'gunshots')
noise_dir = os.path.join(script_dir, 'data', 'noise')
output_dir = os.path.join(script_dir, 'data', 'preprocessed')

print(f"Gunshot directory: {gunshot_dir}")
print(f"Noise directory: {noise_dir}")
print(f"Output directory: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

print("Processing gunshot files:")
preprocess_audio(gunshot_dir, output_dir)

print("\nProcessing noise files:")
preprocess_audio(noise_dir, output_dir)

def extract_features(file_path, sample_rate=16000, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    data, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)  # Take the mean of each coefficient across time
    return mfccs_scaled

def extract_features_from_directory(audio_dir, sample_rate=16000, n_mfcc=13):
    """Extract MFCC features for all files in a directory."""
    features = []
    labels = []
    for filename in os.listdir(audio_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(audio_dir, filename)
            label = 1 if 'gunshot' in filename else 0  # Assuming filenames contain 'gunshot' or 'noise'
            features.append(extract_features(file_path, sample_rate, n_mfcc))
            labels.append(label)
    return np.array(features), np.array(labels)

# Example usage
features, labels = extract_features_from_directory(output_dir)
