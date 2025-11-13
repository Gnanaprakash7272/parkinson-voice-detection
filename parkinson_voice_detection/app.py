import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import joblib
import os
import time
from parkinson_voice_detection.extract_features import extract_features

# 🎨 Page Config
st.set_page_config(page_title="Parkinson's Voice Detection", page_icon="🎤", layout="centered")

# 🌈 Custom CSS
st.markdown("""
<style>
h1, h2, h3, h4 { text-align: center; color: #333333; }
.stButton > button {
    background-color: #4F46E5;
    color: white;
    border-radius: 12px;
    padding: 10px 25px;
    font-size: 16px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: #4338CA;
    transform: scale(1.03);
}
div[data-testid="stSpinner"] {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ✅ Load Model
try:
    model = joblib.load('parkinson_model.pkl')
    st.success("✅ Model loaded successfully!")
except:
    st.error("❌ Model file not found. Please train the model first.")
    model = None

# 🧠 Header Section
st.title("🎤 Parkinson's Disease Voice Risk Assessment")
st.markdown("<p style='text-align:center;'>Analyze your voice and check Parkinson’s risk using AI-powered voice feature extraction.</p>", unsafe_allow_html=True)

# 📜 Instructions (in Card Layout)
with st.container():
    st.markdown("### 🧾 Steps to Follow")
    st.markdown("""
    1️⃣ Click **Start Recording** below  
    2️⃣ Speak clearly for 5 seconds (say: "1 to 10")  
    3️⃣ Wait for the analysis to complete  
    4️⃣ View your **AI-generated risk assessment**
    """)

# 🎙️ Audio Recording
def record_audio(duration=5, sample_rate=22050):
    st.info(f"🎙️ Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()
    return audio.flatten(), sample_rate

# 🎧 Button Centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('🎧 Start Recording', type='primary'):
        if model is None:
            st.error("❌ Model not loaded. Please train the model first.")
        else:
            with st.spinner('🎤 Recording... Please speak now.'):
                audio, sr = record_audio()
                sf.write('temp_audio.wav', audio, sr)
            
            st.success("✅ Recording complete!")
            st.audio('temp_audio.wav', format='audio/wav')

            # 🔬 Feature Extraction
            with st.spinner('🔬 Extracting and analyzing voice features...'):
                time.sleep(2)
                features = extract_features('temp_audio.wav')

                if features is not None:
                    model_features = getattr(model, "n_features_in_", len(features))
                    if len(features) < model_features:
                        missing = model_features - len(features)
                        st.warning(f"⚠️ Model expects {model_features} features. Adding {missing} placeholder(s).")
                        features = np.append(features, [0.0] * missing)
                    elif len(features) > model_features:
                        st.warning(f"⚠️ Trimming extra {len(features) - model_features} features.")
                        features = features[:model_features]

                    # 🧩 Prediction
                    prediction = model.predict([features])
                    probability = model.predict_proba([features])
                    risk_score = probability[0][1]

                    # 📊 Results Display
                    st.subheader("📊 AI Assessment Result")
                    if risk_score < 0.3:
                        st.success(f"🟢 **Low Risk:** {(1 - risk_score)*100:.1f}% confidence of healthy voice.")
                        st.balloons()
                    elif risk_score < 0.7:
                        st.warning(f"🟡 **Moderate Risk:** {risk_score*100:.1f}% probability of Parkinson's.")
                    else:
                        st.error(f"🔴 **High Risk:** {risk_score*100:.1f}% probability of Parkinson's detected.")

                    st.progress(risk_score)
                    st.write(f"**Risk Score:** {risk_score:.3f} (0 = healthy, 1 = Parkinson's)")

                    # 📈 Show Extracted Features
                    with st.expander("📈 View Extracted Features"):
                        feature_names = [
                            'Mean Pitch', 'Max Pitch', 'Min Pitch',
                            'Jitter Local', 'Jitter Absolute', 'Jitter RAP',
                            'Jitter PPQ5', 'Jitter DDP', 'Shimmer Local',
                            'Shimmer dB', 'Shimmer APQ3', 'Shimmer APQ5',
                            'Shimmer DDA', 'HNR', 'Extra Placeholder'
                        ]
                        for name, value in zip(feature_names, features):
                            st.write(f"{name}: {value:.6f}")
                else:
                    st.error("❌ Error in feature extraction. Please record again.")

            # 🧹 Clean temp file
            if os.path.exists('temp_audio.wav'):
                os.remove('temp_audio.wav')

# ⚠️ Disclaimer
st.markdown("---")
st.markdown("""
### ⚠️ Disclaimer  
This is a **research & educational tool**, not a medical diagnostic app.  
Please consult a qualified medical professional for health decisions.
""")

# 🏁 Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center'>
🏆 <b>Built for CSA Hackathon | AI for Early Detection of Parkinson's from Voice</b><br>
👨‍💻 By First Year Student<br>
© 2025 All rights reserved.
</div>
""", unsafe_allow_html=True)
