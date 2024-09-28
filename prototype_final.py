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
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.textlabels import Label

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

# Gemini API setup
GOOGLE_API_KEY = "AIzaSyByVR9XKpvvWDzshhxUKidy3WFFaV--sio"
genai.configure(api_key=GOOGLE_API_KEY)

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

def gemini_analysis(audio_features, beamforming_result, classification_probability):
    """Perform detailed analysis using Gemini AI."""
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    Analyze the following audio data:
    
    1. Audio Features (MFCC): {audio_features.tolist()}
    2. Beamforming Result: Estimated direction {beamforming_result:.2f} degrees
    3. Classification Probability (Gunshot): {classification_probability:.4f}
    
    Based on this data, provide a detailed analysis including:
    1. Likely source of the sound
    2. Confidence in the classification
    3. Potential environmental factors affecting the recording
    4. Recommendations for further analysis or action
    
    Please provide a comprehensive analysis in a clear, structured format.
    """
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
    
    response = model.generate_content(prompt, safety_settings=safety_settings)
    
    return response.text

def create_pdf_report(gemini_result, direction, classification_probability, beamformed_signals, base_dir):
    pdf_file = os.path.join(base_dir, 'gunshot_detection_report.pdf')
    doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    # Instead of adding new styles, modify existing ones
    styles['Heading1'].fontSize = 18
    styles['Heading1'].spaceAfter = 12
    styles['Heading2'].fontSize = 14
    styles['Heading2'].spaceAfter = 8
    styles['BodyText'].fontSize = 12
    styles['BodyText'].spaceAfter = 6

    story = []

    # Title
    story.append(Paragraph("Gunshot Detection Analysis Report", styles['Heading1']))
    story.append(Spacer(1, 12))

    # Date and Time
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['BodyText']))
    story.append(Spacer(1, 12))

    # Summary Table
    data = [
        ["Estimated Direction", f"{direction:.2f} degrees"],
        ["Classification Probability", f"{classification_probability:.4f}"]
    ]
    table = Table(data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Beamforming Plot
    story.append(Paragraph("Beamforming Analysis", styles['Heading2']))
    drawing = Drawing(400, 200)
    lp = LinePlot()
    lp.x = 50
    lp.y = 50
    lp.height = 125
    lp.width = 300
    lp.data = [list(zip(np.linspace(0, 360, len(beamformed_signals)), beamformed_signals))]
    lp.lines[0].strokeColor = colors.blue
    lp.lines[0].strokeWidth = 2
    lp.xValueAxis.valueMin = 0
    lp.xValueAxis.valueMax = 360
    lp.xValueAxis.valueSteps = [0, 90, 180, 270, 360]
    lp.yValueAxis.valueMin = min(beamformed_signals)
    lp.yValueAxis.valueMax = max(beamformed_signals)
    xaxis = Label()
    xaxis.setOrigin(200, 5)
    xaxis.boxAnchor = 'n'
    xaxis.dx = 0
    xaxis.dy = -20
    xaxis.setText('Angle (degrees)')
    yaxis = Label()
    yaxis.setOrigin(10, 100)
    yaxis.angle = 90
    yaxis.boxAnchor = 's'
    yaxis.dx = -20
    yaxis.dy = 0
    yaxis.setText('Signal Strength')
    drawing.add(lp)
    drawing.add(xaxis)
    drawing.add(yaxis)
    story.append(drawing)
    story.append(Spacer(1, 12))

    # Gemini Analysis
    story.append(Paragraph("Detailed AI Analysis", styles['Heading2']))
    for line in gemini_result.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['BodyText']))
        else:
            story.append(Spacer(1, 6))

    doc.build(story)
    print(f"PDF report saved to {pdf_file}")

# Main execution
if __name__ == "__main__":
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

    # Extract features for Gemini analysis
    audio_features = extract_features(recorded_sound)

    # Perform Gemini analysis
    gemini_result = gemini_analysis(audio_features, direction, classification_probability)

    # Create PDF report
    create_pdf_report(gemini_result, direction, classification_probability, beamformed_signals, base_dir)

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

# Save Gemini analysis to a text file
with open(os.path.join(base_dir, 'gemini_analysis.txt'), 'w') as f:
    f.write(gemini_result)

print(f"Gemini analysis saved to {os.path.join(base_dir, 'gemini_analysis.txt')}")