import os
import cv2
import numpy as np
import albumentations as A
from glob import glob

# Define the augmentation pipeline
AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.RandomResizedCrop(160, 160, scale=(0.9, 1.0), p=0.3),
    A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=50, p=0.3),  # Elastic transformation
    A.CoarseDropout(max_holes=1, max_height=8, max_width=8, p=0.3),  # Random Erasing
    A.PerspectiveTransform(scale=(0.01, 0.1), p=0.5)  # Random perspective shift
])

# Directory with input images
INPUT_DIR = "data"
CATEGORIES = ["real", "deepfake", "ai_gen"]
valid_extensions = ['.jpg', '.jpeg', '.png']

for cat in CATEGORIES:
    os.makedirs(os.path.join(INPUT_DIR, cat, 'augmented'), exist_ok=True)  # Create an 'augmented' folder inside each category
    files = glob(f"{INPUT_DIR}/{cat}/*")
    
    for i, file in enumerate(files):
        # Skip non-image files
        if not any(file.lower().endswith(ext) for ext in valid_extensions):
            continue
        
        img = cv2.imread(file)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate 3 augmented images
        for j in range(3):  
            aug = AUG(image=img)["image"]
            save_path = os.path.join(INPUT_DIR, cat, 'augmented', f"aug_{i}_{j}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))

print("✅ Augmentation complete. You can now re-run feature extraction and model training.")
