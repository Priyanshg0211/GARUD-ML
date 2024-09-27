import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import sounddevice as sd
import pywt
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
import sys

# Parameters for simulation
num_mics = 6  # Number of microphones
mic_distance = 0.5  # Distance between microphones (meters)
sound_speed = 343  # Speed of sound (meters per second)
sample_rate = 44100  # Sampling rate (Hz)
lowcut = 300  # Low cutoff frequency for bandpass filter (Hz)
highcut = 5000  # High cutoff frequency for bandpass filter (Hz)
num_angles = 360  # Number of angles to test in beamforming

# Define microphone positions in a circular array
angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
mic_positions = np.column_stack((np.cos(angles), np.sin(angles))) * mic_distance

def record_audio(duration, sample_rate):
    """Record audio from the microphone."""
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print(f"Recording finished. Recorded {len(recording)} samples.")
    return recording.flatten()

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def wavelet_denoising(data, wavelet='haar', level=4):
    """Apply wavelet denoising."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(i, threshold, mode='soft') for i in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def noise_reduction(data, fs):
    """Reduce noise from the audio signal by applying wavelet denoising and bandpass filter."""
    original_max = np.max(np.abs(data))
    denoised_data = wavelet_denoising(data)
    filtered_data = bandpass_filter(denoised_data, lowcut, highcut, fs)
    print(f"Original peak signal amplitude: {original_max}")
    print(f"Post-processing peak signal amplitude: {np.max(np.abs(filtered_data))}")
    return filtered_data

def simulate_microphone_array(mic_positions, recorded_sound, sensitivities):
    """Simulate the sound arrival at each microphone in the array based on their positions."""
    max_delay = int(np.max(np.linalg.norm(mic_positions, axis=1)) / sound_speed * sample_rate)
    output_length = len(recorded_sound) + max_delay
    recordings = np.zeros((num_mics, output_length))
    
    for i, (mic_pos, sensitivity) in enumerate(zip(mic_positions, sensitivities)):
        delay_samples = int(np.linalg.norm(mic_pos) / sound_speed * sample_rate)
        attenuated_signal = noise_reduction(recorded_sound, sample_rate) / (sensitivity ** 2)
        recordings[i, delay_samples:delay_samples + len(recorded_sound)] = attenuated_signal
    
    print(f"Simulated microphone array: {recordings.shape} samples per microphone")
    return recordings

def weighted_delay_and_sum(recordings, mic_positions, weights):
    """Estimate direction of arrival using weighted Delay-and-Sum Beamforming."""
    angle_range = np.linspace(0, 2 * np.pi, num_angles)
    beamformed_signals = np.zeros(num_angles)
    
    for i, angle in enumerate(angle_range):
        delays = np.dot(mic_positions, [np.cos(angle), np.sin(angle)]) / sound_speed * sample_rate
        aligned_signals = np.sum([np.roll(recordings[j], int(round(delays[j]))) * weights[j] for j in range(num_mics)], axis=0)
        beamformed_signals[i] = np.max(np.abs(aligned_signals))
    
    best_angle_index = np.argmax(beamformed_signals)
    best_angle = angle_range[best_angle_index] * 180 / np.pi
    
    print(f"Beamforming Results: Best angle estimate {best_angle:.2f} degrees")
    return best_angle, beamformed_signals

def extract_features(audio_data):
    """Extract MFCC features from the audio data."""
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def classify_sound(model, audio_data):
    """Classify the audio data using the trained model."""
    features = extract_features(audio_data).reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(features)
    return prediction[0][0]

# Create directories for output files with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f'simulation_output_{timestamp}'
png_dir = os.path.join(base_dir, 'png_files')
audio_dir = os.path.join(base_dir, 'audio_files')

os.makedirs(png_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

# Load the trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model', 'gunshot_classifier.keras')
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Record audio
duration = 20  # Duration of recording in seconds
print(f"Please make a sound within the next {duration} seconds...")
recorded_sound = record_audio(duration, sample_rate)
recorded_sound = recorded_sound / np.max(np.abs(recorded_sound))  # Normalize the audio

# Reduce noise
recorded_sound = noise_reduction(recorded_sound, sample_rate)

# Save the recorded audio
audio_file_path = os.path.join(audio_dir, 'recorded_sound.wav')
wavfile.write(audio_file_path, sample_rate, (recorded_sound * 32767).astype(np.int16))
print(f"Saved recorded sound to {audio_file_path}")

# Simulate microphone array processing
sensitivities = np.ones(num_mics)  # Assuming equal sensitivity for simplicity
recordings = simulate_microphone_array(mic_positions, recorded_sound, sensitivities)

# Delay and Sum Beamforming
direction, beamformed_signals = weighted_delay_and_sum(recordings, mic_positions, sensitivities)

# Classify the recorded sound
classification_probability = classify_sound(model, recorded_sound)
print(f"Predicted probability of gunshot: {classification_probability:.4f}")

# Save results to files
np.save(os.path.join(png_dir, 'beamformed_signals.npy'), beamformed_signals)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, 360, num_angles), beamformed_signals)
plt.title('Beamforming Signals vs. Angle')
plt.xlabel('Angle (degrees)')
plt.ylabel('Signal Strength')
plt.grid(True)
plt.savefig(os.path.join(png_dir, 'beamforming_plot.png'))
plt.show()

# Print results
print(f"Estimated Direction: {direction:.2f} degrees")
