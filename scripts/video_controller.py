import cv2
import numpy as np
import os
import time
import argparse
import json

# Import both detectors
# Rename the original file to enhanced_video_prediction.py first
import enhanced_video_prediction
import universal_video

# Optional - import face detection system
try:
    import mediapipe as mp
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # 0 for closer faces, 1 for farther faces
        min_detection_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
    print("[INFO] MediaPipe face detection initialized")
except ImportError:
    print("[WARN] MediaPipe not available, falling back to cascade classifier")
    MEDIAPIPE_AVAILABLE = False
    # Load OpenCV face detector as fallback
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            raise Exception("Failed to load face cascade classifier")
        print("[INFO] OpenCV face detection initialized")
        OPENCV_FACE_AVAILABLE = True
    except Exception as e:
        print(f"[WARN] OpenCV face detection failed: {e}")
        OPENCV_FACE_AVAILABLE = False

def detect_faces_in_video(video_path, max_frames=100):
    """
    Detect if video contains human faces
    Returns: (has_faces, face_proportion)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0 or fps == 0:
        print("[ERROR] Could not read video properties")
        return False, 0.0
    
    # Sample frames at regular intervals
    frames_to_check = min(max_frames, total_frames)
    frame_indices = np.linspace(0, total_frames-1, frames_to_check, dtype=int)
    
    face_frames = 0
    frames_checked = 0
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        frames_checked += 1
        has_face = detect_faces_in_frame(frame)
        if has_face:
            face_frames += 1
    
    cap.release()
    
    # Calculate proportion of frames with faces
    face_proportion = face_frames / frames_checked if frames_checked > 0 else 0
    has_faces = face_proportion > 0.1  # If more than 10% of frames have faces
    
    print(f"[INFO] Face detection: {face_frames}/{frames_checked} frames with faces ({face_proportion:.2f})")
    
    return has_faces, face_proportion

def detect_faces_in_frame(frame):
    """
    Detect faces in a single frame using available detector
    Returns: Boolean
    """
    if MEDIAPIPE_AVAILABLE:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        return results.detections is not None and len(results.detections) > 0
    elif OPENCV_FACE_AVAILABLE:
        # Use OpenCV face detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        return len(faces) > 0
    else:
        # No face detector available
        return False

def analyze_video(video_path, force_detector=None, output_json=None):
    """
    Main function to analyze video
    Detects if video contains human faces and routes to appropriate detector
    
    Parameters:
    - video_path: path to video file
    - force_detector: 'human', 'general', or None (auto-detect)
    - output_json: path to save results as JSON (optional)
    
    Returns: analysis results dict
    """
    start_time = time.time()
    print(f"[INFO] Analyzing video: {video_path}")
    
    # Determine which detector to use
    detector = force_detector
    
    if detector is None:
        # Auto-detect based on content
        has_faces, face_proportion = detect_faces_in_video(video_path)
        
        if has_faces:
            detector = "human"
        else:
            detector = "general"
    
    # Run appropriate detector
    if detector == "human":
        print("[INFO] Using human video detector")
        result = enhanced_video_prediction.predict_video(video_path)
    else:
        print("[INFO] Using general video detector")
        result = universal_video.predict_video(video_path)
    
    # Add metadata to results
    result["detector_used"] = detector
    result["analysis_time"] = time.time() - start_time
    
    # Save results to JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Results saved to {output_json}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect fake videos with appropriate detector")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--force", choices=["human", "general"], 
                        help="Force specific detector (skip auto-detection)")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Run analysis
    result = analyze_video(
        args.video_path, 
        force_detector=args.force,
        output_json=args.output
    )
    
    # Print summary
    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Detector used: {result['detector_used']}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Analysis time: {result['analysis_time']:.2f} seconds")