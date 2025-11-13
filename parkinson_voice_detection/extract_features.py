import parselmouth
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def extract_features(audio_path):
    """
    Extract voice features from audio file for Parkinson's detection
    Returns a list of features compatible with the trained model
    """
    try:
        # Load audio file
        sound = parselmouth.Sound(audio_path)
        
        # Extract pitch
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames

        # Fundamental frequency features
        if len(pitch_values) > 0:
            mean_f0 = np.mean(pitch_values)
            max_f0 = np.max(pitch_values)
            min_f0 = np.min(pitch_values)
        else:
            mean_f0 = max_f0 = min_f0 = 0

        # Create point process for jitter and shimmer analysis
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)

        # Jitter measures (local, absolute, RAP, PPQ5, DDP)
        try:
            jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            jitter_local = 0.0
            
        try:
            jitter_abs = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            jitter_abs = 0.0
            
        try:
            jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            jitter_rap = 0.0
            
        try:
            jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            jitter_ppq5 = 0.0
            
        try:
            jitter_ddp = parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            jitter_ddp = 0.0

        # Shimmer measures
        try:
            shimmer_local = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer_local = 0.0
            
        try:
            shimmer_local_dB = parselmouth.praat.call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer_local_dB = 0.0
            
        try:
            shimmer_apq3 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer_apq3 = 0.0
            
        try:
            shimmer_apq5 = parselmouth.praat.call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer_apq5 = 0.0
            
        try:
            shimmer_dda = parselmouth.praat.call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer_dda = 0.0

        # HNR (Harmonics-to-Noise Ratio)
        try:
            hnr = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            mean_hnr = parselmouth.praat.call(hnr, "Get mean", 0, 0)
        except:
            mean_hnr = 0.0

        # Compile all features
        features = [
            mean_f0, max_f0, min_f0,                    # Pitch features
            jitter_local, jitter_abs, jitter_rap,       # Jitter features
            jitter_ppq5, jitter_ddp,                    # More jitter
            shimmer_local, shimmer_local_dB,            # Shimmer features
            shimmer_apq3, shimmer_apq5, shimmer_dda,    # More shimmer
            mean_hnr                                    # HNR
        ]

        print(f" Successfully extracted {len(features)} voice features")
        return features
        
    except Exception as e:
        print(f" Error in feature extraction: {e}")
        # Return default features if extraction fails
        return [0.0] * 14

# Test function
if __name__ == "__main__":
    print(" Testing feature extraction...")
    # You can add a test audio file here
    print("Feature extraction module ready!")