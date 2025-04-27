import cv2
import torch
import numpy as np
from PIL import Image
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load models
mtcnn = MTCNN(image_size=160, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
clf = joblib.load("model/ensemble_model.pkl")  # Example classifier model
label_map = {0: "real", 1: "deepfake", 2: "ai_gen"}

def extract_faces_from_video(video_path, time_interval_sec=10):
    cap = cv2.VideoCapture(video_path)
    embeddings = []

    # Get the total number of frames in the video and the FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS())
    video_duration = total_frames / fps  # Video duration in seconds
    print(f"[INFO] Video duration: {video_duration} seconds, FPS: {fps}")

    # Calculate the frame skip based on the desired time interval
    frame_skip = int(fps * time_interval_sec)  # Process frames every 'time_interval_sec' seconds
    print(f"[INFO] Processing every {time_interval_sec} seconds. Skipping {frame_skip} frames.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames based on the calculated frame skip
        if frame_idx % frame_skip == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB
            combined_features = extract_combined_features(image)  # Assuming extract_combined_features() is defined
            
            if combined_features is not None:
                embeddings.append(combined_features)

        frame_idx += 1

    cap.release()
    return embeddings

def extract_combined_features(image):
    # Example: Combine features from FaceNet and CLIP (code for this is assumed to be defined already)
    facenet_features = extract_facenet_features(image)
    clip_features = extract_clip_features(image)
    
    if facenet_features is None:
        return None
    
    # Combine (concatenate) the features from FaceNet and CLIP
    combined_features = np.concatenate((facenet_features, clip_features))
    return combined_features

def extract_facenet_features(image):
    # Example function for FaceNet feature extraction
    pass

def extract_clip_features(image):
    # Example function for CLIP feature extraction
    pass

def predict_video(video_path):
    embeddings = extract_faces_from_video(video_path, time_interval_sec=10)
    
    if not embeddings:
        print("[WARN] No faces found in video.")
        return

    # Predict using the classifier
    preds = clf.predict(embeddings)
    
    # Majority voting for final prediction
    final_pred = np.bincount(preds).argmax()  # Most frequent label
    print(f"\n🧠 Final Video Prediction: {label_map[final_pred]} ({len(preds)} frame(s) used)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict_video.py <video_path>")
        sys.exit(1)
    
    # Run the video prediction function
    predict_video(sys.argv[1])
