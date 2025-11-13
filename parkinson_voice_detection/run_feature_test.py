import numpy as np
import soundfile as sf
import os
from extract_features import extract_features

# Generate a 1-second sine wave at 220 Hz and save as WAV
sr = 16000
t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
wave = 0.1 * np.sin(2 * np.pi * 220 * t).astype('float32')

wav_path = os.path.join(os.path.dirname(__file__), 'test_audio.wav')
sf.write(wav_path, wave, sr)
print('WAV written:', wav_path)

# Run feature extraction
features = extract_features(wav_path)
print('Extracted features (len={}):'.format(len(features)))
print(features)
