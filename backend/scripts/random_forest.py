import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# Initialize and train RandomForestClassifier
print("🧠 Training RandomForestClassifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
print("\n📊 Evaluation Report:")
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["real", "deepfake", "ai_gen"]))

# Save the trained model
os.makedirs("model", exist_ok=True)
joblib.dump(rf, "model/random_forest.pkl")

print("\n✅ Model trained and saved to model/random_forest.pkl")
