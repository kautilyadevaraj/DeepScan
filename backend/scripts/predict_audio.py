import sys
import librosa
import numpy as np
import joblib

def extract_mfcc(path):
    try:
        audio, sr = librosa.load(path, sr=16000)  # Load the audio with 16kHz sample rate
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Extract MFCC features
        return np.mean(mfcc.T, axis=0)  # Return the mean of MFCCs across time
    except Exception as e:
        print(f"[ERROR] Failed to process {path}: {e}")
        return None

def predict_audio(path):
    # Extract features from the given audio path
    features = extract_mfcc(path)
    if features is None:
        return  # Exit if feature extraction fails
    
    # Load pre-trained Random Forest model for audio prediction
    model = joblib.load("model/audio_rf.pkl")

    # Reshape the features to match the expected input shape (1, -1)
    features = features.reshape(1, -1)

    # Make prediction
    pred = model.predict(features)[0]
    
    # Map the prediction to class labels (real: 0, fake: 1)
    label = "real" if pred == 0 else "fake"

    # Print the prediction
    print(f"🎧 Prediction: {label}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict_audio.py <audio_path>")
        sys.exit(1)

    # Get the audio file path from command line argument
    path = sys.argv[1]
    
    # Predict the audio label
    predict_audio(path)
