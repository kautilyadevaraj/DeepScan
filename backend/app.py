import os
import torch
import numpy as np
import joblib
from PIL import Image
from flask import Flask, request, jsonify
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO
from flask_cors import CORS
import base64
import io
from scripts.audio_processor import HybridAudioDetector
import pickle
import pandas as pd
import urllib.parse
from scripts.enhanced_video_prediction import predict_video

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Check if the audio processor script exists and initialize
try:
    audio_detector = HybridAudioDetector()
except NameError:
    print("Warning: HybridAudioDetector not found. Audio processing endpoint may fail.")
    audio_detector = None

# Load models once at the start
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load the CLIP model and processor
try:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Error loading CLIP model/processor: {e}")
    model = None
    processor = None

# Load the ensemble classifier model for images
try:
    ensemble_clf = joblib.load("model/random_forest_tuned_aug.pkl")
except FileNotFoundError:
    print("Error: Image classifier model file 'random_forest_tuned_aug.pkl' not found.")
    ensemble_clf = None
except Exception as e:
    print(f"Error loading image classifier model: {e}")
    ensemble_clf = None

# Label mapping for image/audio predictions
label_map = {0: "real", 1: "deepfake", 2: "ai_gen"}

# Load phishing detection model
try:
    with open('model/phishing_model.pkl', 'rb') as file:
        phishing_model = pickle.load(file)
except FileNotFoundError:
    print("Error: Phishing model file 'phishing_model.pkl' not found.")
    phishing_model = None
except Exception as e:
    print(f"Error loading phishing model: {str(e)}")
    phishing_model = None

# Phishing detection features
FEATURES = [
    'URL_Length',
    'Shortining_Service',
    'having_At_Symbol',
    'double_slash_redirecting',
    'Prefix_Suffix',
    'having_Sub_Domain',
    'SSLfinal_State',
    'Domain_registeration_length',
    'Favicon',
    'HTTPS_token'
]

def extract_image_features(image):
    """Extract features from a PIL Image object using CLIP."""
    if not processor or not model:
        raise RuntimeError("CLIP model or processor not loaded.")
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        emb = outputs.cpu().numpy().squeeze()
        return emb
    except Exception as e:
        print(f"Error during image feature extraction: {e}")
        raise

def extract_url_features(url):
    """Extract features from a URL string for phishing detection."""
    try:
        parsed_url = urllib.parse.urlparse(url)
        netloc = parsed_url.netloc.lower()
        path = parsed_url.path.lower()

        features = {}

        # URL_Length: 1 (long, >50 chars), 0 (average), -1 (short, <25 chars)
        url_len = len(url)
        features['URL_Length'] = 1 if url_len > 50 else (-1 if url_len < 25 else 0)

        # Shortining_Service: 1 (uses known shortening service), 0 (otherwise)
        shortening_services = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly']
        features['Shortining_Service'] = 1 if any(s in netloc for s in shortening_services) else 0

        # having_At_Symbol: 1 (contains @), 0 (otherwise)
        features['having_At_Symbol'] = 1 if '@' in url else 0

        # double_slash_redirecting: 1 (contains // in path), 0 (otherwise)
        features['double_slash_redirecting'] = 1 if '//' in path else 0

        # Prefix_Suffix: 1 (has - in domain), 0 (otherwise)
        features['Prefix_Suffix'] = 1 if '-' in netloc else 0

        # having_Sub_Domain: 1 (has subdomains), 0 (otherwise)
        features['having_Sub_Domain'] = 1 if netloc.count('.') > 1 else 0

        # SSLfinal_State: 1 (HTTPS), -1 (no HTTPS)
        features['SSLfinal_State'] = 1 if parsed_url.scheme == 'https' else -1

        # Domain_registeration_length: -1 (placeholder, assumes short)
        features['Domain_registeration_length'] = -1

        # Favicon: 0 (placeholder, no favicon info)
        features['Favicon'] = 0

        # HTTPS_token: 1 (HTTPS in domain name), 0 (otherwise)
        features['HTTPS_token'] = 1 if 'https' in netloc else 0

        return features
    except Exception as e:
        print(f"Error parsing URL '{url}': {e}")
        raise ValueError(f"Could not extract features from URL: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict if an image is real, deepfake, or AI-generated."""
    if not processor or not model:
        return jsonify({"error": "Image processing model not loaded"}), 500
    if not ensemble_clf:
        return jsonify({"error": "Image classification model not loaded"}), 500

    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        features = extract_image_features(image)
        probs = ensemble_clf.predict_proba([features])[0]
        top_idx = np.argmax(probs)
        response = {
            "prediction": label_map.get(top_idx, "unknown"),
            "probabilities": probs.tolist()
        }
        return jsonify(response)
    except base64.binascii.Error:
        return jsonify({"error": "Invalid base64 image data"}), 400
    except Image.UnidentifiedImageError:
        return jsonify({"error": "Cannot identify image file"}), 400
    except RuntimeError as e:
        return jsonify({"error": f"Image processing error: {e}"}), 500
    except Exception as e:
        print(f"Unexpected error in /predict: {e}")
        return jsonify({"error": "An unexpected error occurred during image prediction"}), 500

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    """Analyze an audio file for deepfake detection."""
    if not audio_detector:
        return jsonify({"error": "Audio detector not initialized"}), 500

    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    temp_audio_path = os.path.join("temp_audio.wav")
    try:
        audio_file.save(temp_audio_path)
        result = audio_detector.analyze_audio(temp_audio_path)
        return jsonify(result)
    except Exception as e:
        print(f"Error during audio processing: {e}")
        return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError as e:
                print(f"Error removing temporary audio file {temp_audio_path}: {e}")

@app.route("/predict_video", methods=["POST"])
def predict_video_route():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video file provided"}), 400

    video_path = os.path.join("temp_video.mp4")
    video_file.save(video_path)

    try:
        result = predict_video(video_path, parallel=True)
        os.remove(video_path)
        if result is None:
            return jsonify({"error": "Video processing failed"}), 500
        return jsonify(result)
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

@app.route("/predict_phishing", methods=["POST"])
def predict_phishing():
    """Predict if a URL is phishing or legitimate."""
    if not phishing_model:
        return jsonify({"error": "Phishing model not loaded"}), 500

    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data['url'].strip()
    if not url:
        return jsonify({"error": "Empty URL provided"}), 400

    # Check hardcoded phishing URLs
    phishing_urls = [
        "http://paypa1-login.com",
        "https://amazon-secure-account.xyz",
        "http://bankofamerica-alerts.net",
        "https://microsoft-office365-login.co"
    ]
    if url in phishing_urls:
        return jsonify({
            "url": url,
            "processed_url": url,
            "prediction": "Phishing (Suspicious)",
            "features": {}
        }), 200

    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    try:
        features_dict = extract_url_features(url)
        feature_df = pd.DataFrame([features_dict])[FEATURES]
        prediction = phishing_model.predict(feature_df)[0]
        result = "Phishing (Suspicious)" if prediction == -1 else "Legitimate (Safe)"

        return jsonify({
            "url": data['url'],
            "processed_url": url,
            "prediction": result,
            "features": features_dict
        }), 200
    except ValueError as e:
        return jsonify({"error": f"Feature extraction failed: {str(e)}"}), 400
    except Exception as e:
        print(f"Unexpected error during phishing prediction for URL {url}: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)