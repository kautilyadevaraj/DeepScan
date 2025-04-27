import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Load pre-extracted features and labels
print("📦 Loading pre-extracted features and labels...")

# Load the features (X) and labels (y)
X = np.load("features/embeddings.npy")
y = np.load("features/labels.npy")

print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features each.")

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SVM Classifier
print("🧠 Training SVM Classifier...")
svm = SVC(probability=True, kernel='linear')  # Using probability=True for soft voting
svm.fit(X_train, y_train)

# Evaluate the model
print("\n📊 Evaluation Report:")
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["real", "deepfake", "ai_gen"]))

# Save the trained model
os.makedirs("model", exist_ok=True)
joblib.dump(svm, "model/svm.pkl")

print("\n✅ Model trained and saved to model/svm.pkl")
