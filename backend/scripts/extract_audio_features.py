import os
import librosa
import numpy as np

DATA_DIR = "data"
CATEGORIES = ["real_audio", "fake_audio"]
OUTPUT_DIR = "features_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X, y = [], []

# Data augmentation techniques
def augment_audio(audio, sr):
    # Example: Shift pitch randomly within a range
    pitch_shift = np.random.randint(-5, 5)  # Random pitch shift between -5 and 5 semitones
    audio = librosa.effects.pitch_shift(audio, sr, n_steps=pitch_shift)

    # Example: Time stretch randomly between 0.8x and 1.2x speed
    rate = np.random.uniform(0.8, 1.2)
    audio = librosa.effects.time_stretch(audio, rate)

    # Example: Add random noise
    noise_factor = np.random.uniform(0.001, 0.005)  # Random noise factor
    noise = np.random.randn(len(audio)) * noise_factor
    audio = audio + noise

    return audio

def extract_mfcc(path):
    try:
        audio, sr = librosa.load(path, sr=16000)  # Load the audio with 16kHz sampling rate
        
        # Apply audio augmentation
        audio = augment_audio(audio, sr)
        
        # Extract MFCCs from the augmented audio
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Average MFCC features
        return mfcc_mean
    except Exception as e:
        print(f"[ERROR] Failed to process {path}: {e}")
        return None

# Loop through each category and process audio files
for label, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATA_DIR, cat)
    for fname in os.listdir(folder):
        if not fname.endswith(".wav"):  # Only process .wav files
            continue
        fpath = os.path.join(folder, fname)
        features = extract_mfcc(fpath)  # Extract MFCC features
        if features is not None:
            X.append(features)
            y.append(label)

# Save the extracted features and labels
np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), np.array(X))
np.save(os.path.join(OUTPUT_DIR, "labels.npy"), np.array(y))

print(f"✅ Extracted MFCC features for {len(X)} audio samples.")
