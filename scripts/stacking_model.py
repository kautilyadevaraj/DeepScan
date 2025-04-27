import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# Load the pre-extracted features and labels
print("📦 Loading pre-extracted features and labels...")

# Load the features (X) and labels (y)
X = np.load("features/embeddings.npy")
y = np.load("features/labels.npy")

print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features each.")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, kernel='linear')  # SVM with probability for soft voting
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Create the meta-model (Logistic Regression)
meta_model = LogisticRegression()

# Create the Stacking Classifier
stacking_model = StackingClassifier(estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)], final_estimator=meta_model)

# Train the stacking model
print("🧠 Training the stacking classifier...")
stacking_model.fit(X_train, y_train)

# Evaluate the model
print("\n📊 Evaluation Report:")
y_pred = stacking_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["real", "deepfake", "ai_gen"]))

# Save the trained stacking model
os.makedirs("model", exist_ok=True)
joblib.dump(stacking_model, "model/stacking_model.pkl")

print("\n✅ Stacking model trained and saved to model/stacking_model.pkl")
