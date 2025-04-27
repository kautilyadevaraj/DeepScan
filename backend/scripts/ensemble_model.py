import os
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split

# Load pre-extracted features and labels
print("📦 Loading pre-extracted features and labels...")

# Load the features (X) and labels (y)
X = np.load("features/embeddings.npy")
y = np.load("features/labels.npy")

print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features each.")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize individual classifiers
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, kernel='linear')  # Using probability=True for soft voting
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Create the Voting Classifier ensemble
ensemble_clf = VotingClassifier(estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)], voting='soft')

# Train the ensemble model
print("🧠 Training the ensemble classifier...")
ensemble_clf.fit(X_train, y_train)

# Evaluate the ensemble model
print("\n📊 Evaluation Report:")
y_pred = ensemble_clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["real", "deepfake", "ai_gen"]))

# Save the trained ensemble model
os.makedirs("model", exist_ok=True)
joblib.dump(ensemble_clf, "model/ensemble_model.pkl")

print("\n✅ Ensemble model trained and saved to model/ensemble_model.pkl")
