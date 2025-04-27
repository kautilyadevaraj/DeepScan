import cv2
import torch
import numpy as np
from PIL import Image
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
import librosa
import mediapipe as mp
from scipy.spatial.distance import euclidean
import concurrent.futures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Initialize mediapipe face mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load models
mtcnn = MTCNN(image_size=160, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Try to load model, with a fallback message if it fails
try:
    clf = joblib.load("model/ensemble_model.pkl")
    print("[INFO] Loaded ensemble classifier model")
except:
    print("[WARN] Could not load ensemble model. Will need to train or locate the correct model path.")
    clf = None

label_map = {0: "real", 1: "deepfake", 2: "ai_gen"}

# Lip landmark indices in mediapipe face mesh
LIPS_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]

def extract_audio_features(video_path):
    """Extract audio from video file and compute features"""
    try:
        # Extract audio using librosa
        y, sr = librosa.load(video_path, sr=None)
        
        # Extract audio features (MFCCs are good for voice analysis)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate tempo and beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Create feature vector
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        features = np.concatenate((mfcc_mean, mfcc_std, [tempo]))
        
        return features, y, sr
    except Exception as e:
        print(f"[ERROR] Audio extraction failed: {e}")
        return None, None, None

def extract_lip_movement(frame, face_landmarks):
    """Extract lip movement features from a frame"""
    if face_landmarks is None:
        return None
    
    # Extract lip landmarks
    lip_points = []
    img_h, img_w = frame.shape[:2]
    
    for idx in LIPS_INDICES:
        x = int(face_landmarks.landmark[idx].x * img_w)
        y = int(face_landmarks.landmark[idx].y * img_h)
        lip_points.append((x, y))
    
    # Calculate mouth openness (vertical distance)
    top_lip = lip_points[0]  # Upper lip
    bottom_lip = lip_points[9]  # Lower lip
    mouth_openness = euclidean(top_lip, bottom_lip)
    
    # Calculate mouth width (horizontal distance)
    left_corner = lip_points[5]
    right_corner = lip_points[15]
    mouth_width = euclidean(left_corner, right_corner)
    
    return np.array([mouth_openness, mouth_width])

def analyze_head_motion(face_landmarks_history):
    """Analyze head motion stability and naturalness"""
    if not face_landmarks_history or len(face_landmarks_history) < 10:
        return 0.5  # Default score if not enough data
    
    # Extract nose tip position as a proxy for head position
    nose_positions = []
    for landmarks in face_landmarks_history:
        if landmarks is not None:
            # Use nose tip landmark (index 1)
            nose_positions.append((landmarks.landmark[1].x, landmarks.landmark[1].y))
    
    if len(nose_positions) < 5:
        return 0.5
    
    # Calculate frame-to-frame movement
    movements = []
    for i in range(1, len(nose_positions)):
        prev_pos = nose_positions[i-1]
        curr_pos = nose_positions[i]
        movement = euclidean(prev_pos, curr_pos)
        movements.append(movement)
    
    # Analyze movement patterns
    # Natural motion tends to have some variation but not erratic changes
    mean_movement = np.mean(movements)
    std_movement = np.std(movements)
    
    # Normalize score between 0-1 (higher is more natural)
    # Very low or very high variance can indicate fake videos
    movement_score = np.exp(-(std_movement - 0.01)**2 / 0.0005)
    
    return movement_score

def compute_lip_sync_score(audio, sr, lip_movements, fps):
    """
    Compute a score for lip synchronization between audio and lip movements
    Higher score indicates better sync
    """
    if audio is None or lip_movements is None or len(lip_movements) < 10:
        return 0.5  # Default score if data is missing
    
    # Resample lip movements to match audio timeline
    audio_duration = len(audio) / sr
    video_duration = len(lip_movements) / fps
    
    # Check if we have valid durations
    if audio_duration <= 0 or video_duration <= 0:
        return 0.5
    
    # Get audio amplitude envelope (energy)
    hop_length = 512
    audio_times = librosa.times_like(np.abs(librosa.stft(audio, hop_length=hop_length)), sr=sr)
    audio_energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    
    # Resample lip openness to match audio timeline
    video_times = np.linspace(0, video_duration, len(lip_movements))
    
    try:
        # Only use data within the overlapping time period
        max_time = min(audio_times[-1], video_times[-1])
        min_time = max(audio_times[0], video_times[0])
        
        if min_time >= max_time:
            return 0.5
        
        # Filter both signals to the same time range
        audio_mask = (audio_times >= min_time) & (audio_times <= max_time)
        video_mask = (video_times >= min_time) & (video_times <= max_time)
        
        audio_energy_filtered = audio_energy[audio_mask]
        lip_movements_filtered = [movement[0] for movement in lip_movements[video_mask]]  # Use mouth openness
        
        if len(audio_energy_filtered) < 5 or len(lip_movements_filtered) < 5:
            return 0.5
        
        # Resample lip movements to match audio timeline
        lip_movements_resampled = np.interp(
            audio_times[audio_mask], 
            video_times[video_mask], 
            lip_movements_filtered
        )
        
        # Calculate correlation between audio energy and mouth openness
        correlation = np.corrcoef(audio_energy_filtered, lip_movements_resampled)[0, 1]
        
        # Convert correlation to a 0-1 score (higher is better)
        # A good lip sync should have positive correlation
        sync_score = (correlation + 1) / 2  # Map from [-1,1] to [0,1]
        
        return max(0, min(1, sync_score))  # Ensure score is between 0 and 1
    except Exception as e:
        print(f"[ERROR] Lip sync calculation failed: {e}")
        return 0.5

def process_video_segment(args):
    """Process a segment of video for parallel processing"""
    video_path, start_frame, num_frames, frame_skip = args
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    face_features = []
    lip_movements = []
    face_landmarks_history = []
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process only keyframes
        if _ % frame_skip == 0:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face mesh detection
            results = face_mesh.process(rgb_frame)
            face_landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None
            
            if face_landmarks:
                face_landmarks_history.append(face_landmarks)
                lip_movement = extract_lip_movement(frame, face_landmarks)
                if lip_movement is not None:
                    lip_movements.append(lip_movement)
                
                # Convert to PIL for facenet
                pil_image = Image.fromarray(rgb_frame)
                face = mtcnn(pil_image)
                
                if face is not None:
                    face = face.unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = facenet(face)
                    face_features.append(emb.squeeze().cpu().numpy())
    
    cap.release()
    return face_features, lip_movements, face_landmarks_history

def predict_video(video_path, parallel=True, lip_sync_threshold=0.6):
    """
    Predict if a video is real or deepfake by analyzing facial features,
    lip synchronization with audio, and head motion patterns
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if total_frames == 0 or fps == 0:
        print("[ERROR] Could not read video properties")
        return
    
    print(f"[INFO] Video has {total_frames} frames at {fps} FPS")
    
    # Extract audio features
    audio_features, audio, sr = extract_audio_features(video_path)
    if audio_features is None:
        print("[WARN] Could not extract audio features")
    else:
        print(f"[INFO] Extracted {len(audio_features)} audio features")
    
    # Define frame skip rate (analyze every N frames)
    frame_skip = max(1, int(fps / 4))  # Analyze ~4 frames per second
    
    face_features = []
    lip_movements = []
    face_landmarks_history = []
    
    if parallel and total_frames > 100:
        # Process video in parallel for large videos
        segment_size = 100  # frames per segment
        num_segments = total_frames // segment_size + 1
        
        args_list = [
            (video_path, i * segment_size, segment_size, frame_skip) 
            for i in range(num_segments)
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_video_segment, args_list))
            
        for segment_face_features, segment_lip_movements, segment_landmarks in results:
            face_features.extend(segment_face_features)
            lip_movements.extend(segment_lip_movements)
            face_landmarks_history.extend(segment_landmarks)
    else:
        # Process video sequentially
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process only keyframes
            if frame_idx % frame_skip == 0:
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face mesh detection
                results = face_mesh.process(rgb_frame)
                face_landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None
                
                if face_landmarks:
                    face_landmarks_history.append(face_landmarks)
                    lip_movement = extract_lip_movement(frame, face_landmarks)
                    if lip_movement is not None:
                        lip_movements.append(lip_movement)
                    
                    # Convert to PIL for facenet
                    pil_image = Image.fromarray(rgb_frame)
                    face = mtcnn(pil_image)
                    
                    if face is not None:
                        face = face.unsqueeze(0).to(device)
                        with torch.no_grad():
                            emb = facenet(face)
                        face_features.append(emb.squeeze().cpu().numpy())
            
            frame_idx += 1
        
        cap.release()
    
    print(f"[INFO] Processed {len(face_features)} faces")
    
    if not face_features:
        print("[WARN] No faces detected in video")
        return
    
    # Analyze lip sync
    lip_sync_score = compute_lip_sync_score(audio, sr, np.array(lip_movements) if lip_movements else None, fps)
    print(f"[INFO] Lip sync score: {lip_sync_score:.4f} (higher is better)")
    
    # Analyze head motion
    head_motion_score = analyze_head_motion(face_landmarks_history)
    print(f"[INFO] Head motion naturalness score: {head_motion_score:.4f} (higher is better)")
    
    # Base predictions on facial features if classifier is available
    deepfake_probability = 0.5  # Default probability
    
    if clf is not None and face_features:
        # Get predictions for each face
        face_preds = []
        face_probs = []
        
        for emb in face_features:
            pred = clf.predict([emb])[0]
            probs = clf.predict_proba([emb])[0]
            face_preds.append(pred)
            face_probs.append(probs)
        
        # Majority vote for classification
        final_pred = np.bincount(face_preds).argmax()
        
        # Average probabilities
        avg_probs = np.mean(face_probs, axis=0)
        deepfake_probability = avg_probs[1] if len(avg_probs) > 1 else 0.5
        
        print(f"\n🧠 Base model prediction: {label_map[final_pred]} (confidence: {avg_probs[final_pred]:.4f})")
        print("\n📊 Class probabilities:")
        for i, prob in enumerate(avg_probs):
            print(f"  - {label_map[i]}: {prob:.4f}")
    
    # Adjust the final prediction based on lip sync and head motion analysis
    # A sophisticated deepfake might fool the image classifier but have poor lip sync or unnatural head motion
    
    # Low lip sync score or unnatural head motion increases deepfake probability
    if lip_sync_score < lip_sync_threshold:
        print(f"[WARN] Poor lip synchronization detected ({lip_sync_score:.4f} < {lip_sync_threshold})")
        deepfake_probability = max(deepfake_probability, 0.7)  # Increase deepfake probability
    
    if head_motion_score < 0.4:
        print(f"[WARN] Unnatural head motion detected ({head_motion_score:.4f} < 0.4)")
        deepfake_probability = max(deepfake_probability, 0.65)  # Increase deepfake probability
    
    # Make final decision
    final_label = "deepfake" if deepfake_probability > 0.5 else "real"
    confidence = max(deepfake_probability, 1 - deepfake_probability)
    
    print(f"\n🎯 FINAL VERDICT: {final_label.upper()} (confidence: {confidence:.4f})")
    print(f"   - Based on visual features, lip sync ({lip_sync_score:.2f}), and head motion ({head_motion_score:.2f})")
    
    result = {
        "prediction": final_label,
        "confidence": float(confidence),
        "lip_sync_score": float(lip_sync_score),
        "head_motion_score": float(head_motion_score)
    }
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict_video.py <video_path>")
        sys.exit(1)
    
    # Run the video prediction function
    predict_video(sys.argv[1])