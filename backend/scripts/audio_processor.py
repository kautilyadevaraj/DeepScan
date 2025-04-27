
import librosa
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import scipy
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class HybridAudioDetector:
    def __init__(self):
        # Initialize ML model
        self.feature_processor = AutoFeatureExtractor.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")  # Correct model path
        self.ml_classifier = AutoModelForAudioClassification.from_pretrained("MelodyMachine/Deepfake-audio-detection-V2")  # Correct model path
        
        # Heuristic analysis parameters
        self.sample_rate = 16000
        self.min_duration = 1.0  # Minimum audio duration in seconds
        
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
            if len(waveform.shape) > 1:  # Convert stereo to mono
                waveform = np.mean(waveform, axis=1)
            return waveform, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def model_prediction(self, waveform):
        """Get prediction from ML classifier"""
        inputs = self.feature_processor(
            waveform, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.ml_classifier(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return {
            "prediction": "real" if probabilities[0][0] > probabilities[0][1] else "fake",
            "real_prob": probabilities[0][0].item(),
            "fake_prob": probabilities[0][1].item()
        }
    
    def heuristic_analysis(self, waveform, sr):
        """Perform heuristic audio analysis"""
        # Basic audio features
        rms = librosa.feature.rms(y=waveform)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        
        # Calculate heuristic scoresz
        scores = {
            "volume_variation": np.std(rms) / (np.mean(rms) if np.mean(rms) > 0 else 1),
            "zcr_score": np.mean(zcr),
            "spectral_variation": np.std(spectral_centroid) / (np.mean(spectral_centroid) if np.mean(spectral_centroid) > 0 else 1),
            "formant_score": self.analyze_formant_consistency(waveform, sr),
            "splice_score": self.detect_audio_splices(waveform, sr),
            "noise_score": self.analyze_background_noise(waveform, sr)
        }
        
        # Combine heuristic scores
        combined_score = (
            scores["formant_score"] * 0.4 +
            scores["splice_score"] * 0.3 +
            scores["noise_score"] * 0.2 +
            (1 - scores["spectral_variation"]) * 0.1  # Lower variation is suspicious
        )
        
        return {
            "prediction": "fake" if combined_score > 0.5 else "real",
            "confidence": max(combined_score, 1 - combined_score),
            "scores": scores
        }
    
    def analyze_formant_consistency(self, y, sr):
        """Analyze formant consistency (real voices have natural variations)"""
        if len(y) < sr * 1:  # Need at least 1 second
            return 0.5
        
        try:
            frame_length = 512
            hop_length = 256
            formant_stabilities = []
            
            for i in range(0, len(y) - frame_length, hop_length):
                frame = y[i:i + frame_length]
                lpc_coeffs = librosa.lpc(frame, order=12)
                roots = np.polynomial.polynomial.polyroots(np.flip(lpc_coeffs))
                roots = roots[np.imag(roots) > 0]
                roots = roots[np.abs(roots) < 0.99]
                
                if len(roots) > 0:
                    angles = np.angle(roots)
                    formants = angles * sr / (2 * np.pi)
                    formants = np.sort(formants)[:3] if len(formants) >= 3 else formants
                    formant_stabilities.append(formants)
            
            if not formant_stabilities or len(formant_stabilities) < 3:
                return 0.5
                
            formant_stabilities = np.array(formant_stabilities)
            stds = []
            means = []
            for i in range(min(3, formant_stabilities.shape[1])):
                stds.append(np.std(formant_stabilities[:, i]))
                means.append(np.mean(formant_stabilities[:, i]))
            
            cvs = []
            for i in range(len(means)):
                if means[i] > 0:
                    cvs.append(stds[i] / means[i])
                    
            if not cvs:
                return 0.5
                    
            avg_cv = np.mean(cvs)
            
            if avg_cv < 0.05: return 0.8
            elif avg_cv < 0.1: return 0.6
            else: return 0.3
                
        except Exception as e:
            print(f"Formant analysis error: {e}")
            return 0.5
    
    def detect_audio_splices(self, y, sr):
        """Detect potential audio splices/edits"""
        if len(y) < sr * 3:  # Need at least 3 seconds
            return 0.5
        
        try:
            S = np.abs(librosa.stft(y))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            delta = librosa.feature.delta(S_db)
            delta_sum = np.sum(np.abs(delta), axis=0)
            delta_norm = delta_sum / np.mean(delta_sum)
            peaks = scipy.signal.find_peaks(delta_norm, height=3.0, distance=sr//512)[0]
            
            if len(peaks) == 0:
                return 0.2
            
            peak_heights = delta_norm[peaks]
            duration = len(y) / sr
            peak_density = len(peaks) / duration
            peak_height_mean = np.mean(peak_heights)
            splice_score = min(0.3 * peak_density + 0.7 * peak_height_mean / 5.0, 1.0)
            return splice_score
            
        except Exception as e:
            print(f"Splice detection error: {e}")
            return 0.5
    
    def analyze_background_noise(self, y, sr):
        """Analyze background noise characteristics"""
        try:
            rms = librosa.feature.rms(y=y)[0]
            rms_threshold = np.mean(rms) * 0.5
            quiet_mask = rms < rms_threshold
            
            if not np.any(quiet_mask):
                return 0.5
                
            hop_length = 512
            noise_features = []
            
            in_section = False
            start_idx = 0
            for i, is_quiet in enumerate(quiet_mask):
                if is_quiet and not in_section:
                    in_section = True
                    start_idx = i
                elif not is_quiet and in_section:
                    in_section = False
                    if i - start_idx > 5:
                        sample_start = start_idx * hop_length
                        sample_end = min(i * hop_length, len(y))
                        noise_segment = y[sample_start:sample_end]
                        
                        if len(noise_segment) >= sr // 4:
                            noise_spec = np.abs(librosa.stft(noise_segment))
                            noise_db = librosa.amplitude_to_db(noise_spec, ref=np.max)
                            mean_per_freq = np.mean(noise_db, axis=1)
                            std_per_freq = np.std(noise_db, axis=1)
                            flatness = np.mean(std_per_freq) / (np.std(mean_per_freq) if np.std(mean_per_freq) > 0 else 1)
                            noise_features.append(flatness)
            
            if in_section and len(quiet_mask) - start_idx > 5:
                sample_start = start_idx * hop_length
                sample_end = min(len(quiet_mask) * hop_length, len(y))
                noise_segment = y[sample_start:sample_end]
                
                if len(noise_segment) >= sr // 4:
                    noise_spec = np.abs(librosa.stft(noise_segment))
                    noise_db = librosa.amplitude_to_db(noise_spec, ref=np.max)
                    mean_per_freq = np.mean(noise_db, axis=1)
                    std_per_freq = np.std(noise_db, axis=1)
                    flatness = np.mean(std_per_freq) / (np.std(mean_per_freq) if np.std(mean_per_freq) > 0 else 1)
                    noise_features.append(flatness)
            
            if not noise_features:
                return 0.5
                
            avg_flatness = np.mean(noise_features)
            
            if avg_flatness < 0.5: return 0.7
            elif avg_flatness > 2.0: return 0.6
            else: return 0.3
                
        except Exception as e:
            print(f"Noise analysis error: {e}")
            return 0.5
    
    def analyze_audio(self, audio_path):
        """Main analysis function combining both approaches"""
        waveform, sr = self.load_audio(audio_path)
        if waveform is None:
            return {"error": "Could not load audio file"}
        
        if len(waveform) / sr < self.min_duration:
            return {"error": f"Audio too short (min {self.min_duration}s required)"}
        
        # Get model prediction
        model_result = self.model_prediction(waveform)
        
        # Get heuristic analysis
        heuristic_result = self.heuristic_analysis(waveform, sr)
        
        # Combine results with weighting
        model_weight = 0.6  # Higher weight for model
        heuristic_weight = 0.4
        
        if model_result["prediction"] == "real":
            combined_prob = (model_result["real_prob"] * model_weight + 
                           (1 - heuristic_result["confidence"] if heuristic_result["prediction"] == "fake" else heuristic_result["confidence"]) * heuristic_weight)
            final_pred = "real" if combined_prob > 0.5 else "fake"
        else:
            combined_prob = (model_result["fake_prob"] * model_weight + 
                           (heuristic_result["confidence"] if heuristic_result["prediction"] == "fake" else 1 - heuristic_result["confidence"]) * heuristic_weight)
            final_pred = "fake" if combined_prob > 0.5 else "real"
        
        confidence = max(combined_prob, 1 - combined_prob)
        


        return {
                "final_prediction": final_pred,
                "final_confidence": float(confidence),  # Convert to float
                "model_prediction": model_result["prediction"],
                "model_confidence": float(max(model_result["real_prob"], model_result["fake_prob"])),  # Convert to float
                "heuristic_prediction": heuristic_result["prediction"],
                "heuristic_confidence": float(heuristic_result["confidence"]),  # Convert to float
                "heuristic_scores": {k: float(v) for k, v in heuristic_result["scores"].items()}  # Convert scores to float
            }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Audio Detector with ML")
    parser.add_argument("audio_path", help="Path to audio file for analysis")
    args = parser.parse_args()
    
    detector = HybridAudioDetector()
    result = detector.analyze_audio(args.audio_path)
    
    print("\n=== Analysis Results ===")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Final Prediction: {result['final_prediction'].upper()} (Confidence: {result['final_confidence']:.2f})")
        print(f"\nModel Prediction: {result['model_prediction'].upper()} (Confidence: {result['model_confidence']:.2f})")
        print(f"Heuristic Prediction: {result['heuristic_prediction'].upper()} (Confidence: {result['heuristic_confidence']:.2f})")
        
        print("\nAnalysis Scores:")
        for k, v in result['heuristic_scores'].items():
            print(f"- {k.replace('_', ' ').title()}: {v:.4f}")