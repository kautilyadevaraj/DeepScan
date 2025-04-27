import sys
import torch
import numpy as np
import joblib
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
ensemble_clf = joblib.load("models/random_forest_aug.pkl")

label_map = {0: "real", 1: "deepfake", 2: "ai_gen"}

def extract_features(image):
    image = image.resize((224, 224))  # Resize image
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    emb = outputs.cpu().numpy().squeeze()
    return emb

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    features = extract_features(image)
    probs = ensemble_clf.predict_proba([features])[0]
    top_idx = np.argmax(probs)
    print(f"Prediction: {label_map[top_idx]}")
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])


