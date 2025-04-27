import cv2
import torch
import numpy as np
from PIL import Image
import joblib
from torchvision import models, transforms
import librosa
from scipy.spatial.distance import euclidean
import concurrent.futures
import os
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load general feature extractor
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model = model.eval().to(device)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Try to load classifier model if available
try:
    clf = joblib.load("model/general_video_detector.pkl")
    print("[INFO] Loaded general video classifier model")
except:
    print("[WARN] Could not load general video model. Will use heuristic detection only.")
    clf = None

label_map = {0: "real", 1: "fake"}

def extract_audio_features(video_path):
    """Extract audio from video file and compute features"""
    try:
        # Extract audio using librosa
        y, sr = librosa.load(video_path, sr=None)
        
        # Extract audio features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Calculate tempo and beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Create feature vector
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        contrast_mean = np.mean(spectral_contrast, axis=1)
        
        features = np.concatenate((mfcc_mean, mfcc_std, contrast_mean, [tempo]))
        
        return features, y, sr
    except Exception as e:
        print(f"[ERROR] Audio extraction failed: {e}")
        return None, None, None

def detect_temporal_inconsistencies(frame_features):
    """
    Detect unnatural changes between frames
    Higher scores indicate potential manipulation
    """
    if len(frame_features) < 10:
        return 0.5  # Not enough data
    
    # Calculate frame-to-frame differences
    diffs = []
    for i in range(1, len(frame_features)):
        diff = np.linalg.norm(frame_features[i] - frame_features[i-1])
        diffs.append(diff)
    
    # Analyze the pattern of differences
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    
    # Look for unusually abrupt changes (spikes) which may indicate manipulation
    spikes = [diff for diff in diffs if diff > mean_diff + 2 * std_diff]
    spike_ratio = len(spikes) / len(diffs) if diffs else 0
    
    # Also check for unusually consistent motion (too smooth, like in some AI videos)
    smoothness = 1.0 - (std_diff / mean_diff) if mean_diff > 0 else 0.5
    
    # Combine metrics - higher score means more likely to be manipulated
    inconsistency_score = (spike_ratio * 0.7) + (smoothness * 0.3)
    
    # Normalize to 0-1 range
    return min(max(inconsistency_score, 0), 1)

def analyze_optical_flow(frames, sample_rate=5):
    """
    Analyze optical flow patterns across video frames
    Many AI-generated videos have unnatural motion patterns
    """
    if len(frames) < 6:
        return 0.5  # Not enough data
    
    flow_scores = []
    prev_gray = None
    
    for i in range(0, len(frames), sample_rate):
        if i == 0:
            if len(frames[i].shape) == 3:
                prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = frames[i]
            continue
            
        if len(frames[i].shape) == 3:
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        else:
            gray = frames[i]
            
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Analyze magnitude and direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Statistical analysis of flow
        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)
        angle_std = np.std(angle)
        
        # Calculate flow consistency (too consistent = suspiciously smooth)
        if mag_mean > 0:
            mag_variation = mag_std / mag_mean
        else:
            mag_variation = 1.0
            
        # Calculate direction consistency
        direction_consistency = 1.0 - min(angle_std / np.pi, 1.0)
        
        # Unnatural motion often has too consistent direction and magnitude
        # Score higher if motion is too smooth or too uniform
        if mag_variation < 0.3 and direction_consistency > 0.7:
            score = 0.8  # Very suspicious - too uniform motion
        elif mag_variation < 0.5 and direction_consistency > 0.6:
            score = 0.6  # Somewhat suspicious motion
        else:
            score = 0.3  # More natural motion patterns
            
        flow_scores.append(score)
        prev_gray = gray
    
    return np.mean(flow_scores) if flow_scores else 0.5

def analyze_noise_patterns(frames, sample_rate=10):
    """
    Analyze noise patterns across video frames
    Many AI-generated videos have distinctive noise signatures
    """
    if len(frames) < 5:
        return 0.5  # Not enough data
    
    # Sample frames to reduce computation
    sampled_frames = frames[::sample_rate]
    
    # Extract noise from each frame
    noise_patterns = []
    for frame in sampled_frames:
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Denoise
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Extract noise
        noise = cv2.absdiff(gray, denoised)
        
        # Calculate noise statistics
        noise_mean = np.mean(noise)
        noise_std = np.std(noise)
        
        # Calculate frequency domain properties
        f_transform = np.fft.fft2(noise)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Get properties of frequency spectrum
        f_mean = np.mean(magnitude)
        f_std = np.std(magnitude)
        
        noise_patterns.append((noise_mean, noise_std, f_mean, f_std))
    
    # Analyze consistency of noise patterns
    noise_means = [p[0] for p in noise_patterns]
    noise_stds = [p[1] for p in noise_patterns]
    f_means = [p[2] for p in noise_patterns]
    f_stds = [p[3] for p in noise_patterns]
    
    # Calculate variability metrics
    spatial_consistency = np.std(noise_means) / np.mean(noise_means) if np.mean(noise_means) > 0 else 0
    spectral_consistency = np.std(f_means) / np.mean(f_means) if np.mean(f_means) > 0 else 0
    
    # AI-generated content often has too consistent or structured noise
    if spatial_consistency < 0.1 or spectral_consistency < 0.1:
        # Too uniform - suspicious
        return 0.8
    elif spatial_consistency > 0.6 or spectral_consistency > 0.6:
        # Too erratic - also suspicious
        return 0.7
    else:
        # Natural variation
        return 0.3

def detect_compression_artifacts(frames):
    """
    Detect unusual compression artifacts that might indicate manipulation
    """
    if len(frames) < 5:
        return 0.5
    
    artifact_scores = []
    
    for frame in frames:
        # Convert to YCrCb color space
        if len(frame.shape) == 3:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            # Focus on luminance channel for general analysis
            y_channel = ycrcb[:,:,0]
            
            # Detect blockiness (common in compressed fakes)
            sobelx = cv2.Sobel(y_channel, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(y_channel, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitudes
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Look for grid patterns - split into blocks and check edges
            h, w = y_channel.shape
            block_size = 8  # Standard JPEG/MPEG block size
            
            edge_strengths = []
            
            # Check horizontal edges
            for y in range(block_size, h, block_size):
                # Get the edge strength along this horizontal line
                edge_strength = np.mean(magnitude[y-1:y+1, :])
                edge_strengths.append(edge_strength)
                
            # Check vertical edges
            for x in range(block_size, w, block_size):
                # Get the edge strength along this vertical line
                edge_strength = np.mean(magnitude[:, x-1:x+1])
                edge_strengths.append(edge_strength)
                
            if not edge_strengths:
                continue
                
            # Calculate mean and std of edge strengths
            mean_edge = np.mean(edge_strengths)
            std_edge = np.std(edge_strengths)
            
            # Calculate non-edge gradient mean for comparison
            non_edge_mean = np.mean(magnitude)
            
            # Blockiness ratio (how much stronger are edges at block boundaries?)
            if non_edge_mean > 0:
                blockiness = mean_edge / non_edge_mean
            else:
                blockiness = 1.0
                
            # Consistency of block edges (too consistent is suspicious)
            edge_consistency = 1.0 - min(std_edge / mean_edge, 1.0) if mean_edge > 0 else 0.5
            
            # Combine scores - higher means more likely to be manipulated
            artifact_score = (blockiness * 0.7 + edge_consistency * 0.3) / 2
            artifact_scores.append(min(artifact_score, 1.0))
    
    # Return the average artifact score
    return np.mean(artifact_scores) if artifact_scores else 0.5

def analyze_color_distribution(frames, sample_rate=15):
    """
    Analyze color distribution across video frames
    AI-generated videos often have unnatural color patterns
    """
    if len(frames) < 5:
        return 0.5
        
    sampled_frames = frames[::sample_rate]
    
    color_scores = []
    
    for frame in sampled_frames:
        if len(frame.shape) != 3:
            continue
            
        # Analyze RGB channels
        b, g, r = cv2.split(frame)
        
        # Calculate histograms
        hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
        
        # Normalize histograms
        hist_r = cv2.normalize(hist_r, hist_r).flatten()
        hist_g = cv2.normalize(hist_g, hist_g).flatten()
        hist_b = cv2.normalize(hist_b, hist_b).flatten()
        
        # Calculate entropy (measure of information/randomness)
        entropy_r = -np.sum(hist_r * np.log2(hist_r + 1e-7))
        entropy_g = -np.sum(hist_g * np.log2(hist_g + 1e-7))
        entropy_b = -np.sum(hist_b * np.log2(hist_b + 1e-7))
        
        # Calculate average entropy
        avg_entropy = (entropy_r + entropy_g + entropy_b) / 3
        
        # AI images often have different color entropy patterns
        # Score based on entropy - too low or too high is suspicious
        if avg_entropy < 3.0:
            score = 0.7  # Too little entropy - suspicious
        elif avg_entropy > 7.5:
            score = 0.6  # Too much entropy - somewhat suspicious
        else:
            score = 0.3  # Natural range
            
        color_scores.append(score)
    
    return np.mean(color_scores) if color_scores else 0.5

def extract_frame_features(frame):
    """Extract deep features from a frame using ResNet"""
    try:
        # Convert to PIL and preprocess
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(input_batch)
        
        # Return as numpy array
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None

def process_video_segment(args):
    """Process a segment of video for parallel processing"""
    video_path, start_frame, num_frames, frame_skip = args
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_features = []
    frames = []
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process only keyframes
        if _ % frame_skip == 0:
            frames.append(frame)
            features = extract_frame_features(frame)
            if features is not None:
                frame_features.append(features)
    
    cap.release()
    return frame_features, frames

def predict_video(video_path, parallel=True):
    """
    Predict if a video is real or fake by analyzing general video features,
    temporal consistency, optical flow, and noise patterns
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if total_frames == 0 or fps == 0:
        print("[ERROR] Could not read video properties")
        return {"prediction": "unknown", "confidence": 0.5}
    
    print(f"[INFO] Video has {total_frames} frames at {fps} FPS ({width}x{height})")
    
    # Extract audio features
    audio_features, audio, sr = extract_audio_features(video_path)
    if audio_features is None:
        print("[WARN] Could not extract audio features")
    else:
        print(f"[INFO] Extracted {len(audio_features)} audio features")
    
    # Define frame skip rate (analyze every N frames)
    frame_skip = max(1, int(fps / 2))  # Analyze ~2 frames per second for efficiency
    
    frame_features = []
    all_frames = []
    
    # Process time measurement
    start_time = time.time()
    
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
            
        for segment_features, segment_frames in results:
            frame_features.extend(segment_features)
            all_frames.extend(segment_frames)
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
                all_frames.append(frame)
                features = extract_frame_features(frame)
                if features is not None:
                    frame_features.append(features)
            
            frame_idx += 1
        
        cap.release()
    
    process_time = time.time() - start_time
    print(f"[INFO] Processed {len(frame_features)} frames in {process_time:.2f} seconds")
    
    if not frame_features:
        print("[WARN] No features extracted from video")
        return {"prediction": "unknown", "confidence": 0.5}
    
    # Analyze temporal inconsistencies
    temporal_score = detect_temporal_inconsistencies(frame_features)
    print(f"[INFO] Temporal inconsistency score: {temporal_score:.4f} (higher indicates potential fake)")
    
    # Analyze optical flow
    flow_score = analyze_optical_flow(all_frames)
    print(f"[INFO] Optical flow analysis score: {flow_score:.4f} (higher indicates potential fake)")
    
    # Analyze noise patterns
    noise_score = analyze_noise_patterns(all_frames)
    print(f"[INFO] Noise pattern analysis score: {noise_score:.4f} (higher indicates potential fake)")
    
    # Detect compression artifacts
    artifact_score = detect_compression_artifacts(all_frames[:20])  # Sample first 20 frames
    print(f"[INFO] Compression artifact score: {artifact_score:.4f} (higher indicates potential fake)")
    
    # Analyze color distribution
    color_score = analyze_color_distribution(all_frames)
    print(f"[INFO] Color distribution score: {color_score:.4f} (higher indicates potential fake)")
    
    # Initialize fake probability
    fake_probability = 0.5  # Default
    
    # Use classifier if available
    if clf is not None:
        try:
            # Use average feature vector for classification
            avg_features = np.mean(frame_features, axis=0)
            pred = clf.predict([avg_features])[0]
            probs = clf.predict_proba([avg_features])[0]
            
            print(f"\n🧠 Base model prediction: {label_map[pred]} (confidence: {probs[pred]:.4f})")
            fake_probability = probs[1]  # Probability of being fake
        except Exception as e:
            print(f"[ERROR] Classifier prediction failed: {e}")
    
    # Combine heuristic scores if no classifier or as additional signal
    # Weight the different analysis methods
    combined_heuristic_score = (
        temporal_score * 0.25 +
        flow_score * 0.25 +
        noise_score * 0.2 +
        artifact_score * 0.15 +
        color_score * 0.15
    )
    
    print(f"[INFO] Combined heuristic score: {combined_heuristic_score:.4f}")
    
    # Final decision - blend model prediction with heuristics
    # If we have a classifier, use it as base and adjust with heuristics
    # Otherwise rely entirely on heuristics
    if clf is not None:
        # Adjust model prediction with heuristic insights
        final_fake_probability = fake_probability * 0.7 + combined_heuristic_score * 0.3
    else:
        # Use only heuristics
        final_fake_probability = combined_heuristic_score
    
    # Make final prediction
    final_label = "fake" if final_fake_probability > 0.5 else "real"
    confidence = max(final_fake_probability, 1 - final_fake_probability)
    
    print(f"\n🎯 FINAL VERDICT: {final_label.upper()} (confidence: {confidence:.4f})")
    print(f"   - Based on visual features, temporal patterns, and statistical analysis")
    
    result = {
        "prediction": final_label,
        "confidence": float(confidence),
        "temporal_score": float(temporal_score),
        "flow_score": float(flow_score),
        "noise_score": float(noise_score),
        "artifact_score": float(artifact_score),
        "color_score": float(color_score)
    }
    
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python general_video_detector.py <video_path>")
        sys.exit(1)
    
    # Run the video prediction function
    predict_video(sys.argv[1])