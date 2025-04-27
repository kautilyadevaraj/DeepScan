import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
from transformers import CLIPProcessor, CLIPModel
import albumentations as A
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Initialize models
mtcnn = MTCNN(image_size=160, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Input data folders
DATA_DIR = "data"
CATEGORIES = ["real", "deepfake", "ai_gen"]

# Output path
os.makedirs("features", exist_ok=True)

# Data augmentation pipeline
augment = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.3),
    A.MotionBlur(p=0.2),
    A.Resize(160, 160),  # For MTCNN size requirement
])

def extract_facenet_features(img_path):
    image = Image.open(img_path).convert("RGB")
    
    # Resize image before passing it to MTCNN
    img_np = np.array(image)
    img_resized = cv2.resize(img_np, (160, 160))  # Resize image to 160x160

    # Apply augmentation
    augmented = augment(image=img_resized)["image"]
    img_aug = Image.fromarray(augmented)

    # Face detection using MTCNN
    face = mtcnn(img_aug)
    if face is None:
        print(f"[WARN] No face detected in {img_path}")
        return None
    face = face.unsqueeze(0).to(device)

    # Feature extraction using FaceNet
    with torch.no_grad():
        face_emb = facenet(face)
    
    return face_emb.squeeze().cpu().numpy()

def extract_clip_features(img_path):
    image = Image.open(img_path).convert("RGB")
    
    # Apply the same augmentation to the image before passing to CLIP
    img_np = np.array(image)
    augmented = augment(image=img_np)["image"]
    img_aug = Image.fromarray(augmented)

    # Extract features using CLIP
    inputs = clip_processor(images=img_aug, return_tensors="pt").to(device)
    with torch.no_grad():
        clip_outputs = clip_model.get_image_features(**inputs)
    
    return clip_outputs.cpu().numpy().squeeze()

def extract_combined_features(img_path):
    # Extract features from both FaceNet and CLIP
    facenet_features = extract_facenet_features(img_path)
    clip_features = extract_clip_features(img_path)
    
    if facenet_features is None:
        return None
    
    # Combine (concatenate) the features from FaceNet and CLIP
    combined_features = np.concatenate((facenet_features, clip_features))
    return combined_features

def extract_all_features():
    X, y = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, category)
        if not os.path.isdir(folder):
            print(f"[WARN] Missing folder: {folder}")
            continue

        print(f"\n🧠 Extracting from: {category} ({folder})")
        for fname in tqdm(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, fname)
            combined_features = extract_combined_features(path)
            if combined_features is not None:
                X.append(combined_features)
                y.append(label)

    # Save the extracted features
    np.save("../features/embeddings.npy", np.array(X))
    np.save("../features/labels.npy", np.array(y))
    print(f"\n✅ Done: Saved {len(X)} embeddings.")

if __name__ == "__main__":
    extract_all_features()
