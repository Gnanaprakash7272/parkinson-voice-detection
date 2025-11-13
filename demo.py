import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import joblib
import os
import time
from parkinson_voice_detection.extract_features import extract_features

# Set Streamlit page config (must be first Streamlit command)
st.set_page_config(page_title="Parkinson's Disease Detection", page_icon="🎤")

# Load the trained model
try:
    model = joblib.load('parkinson_model.pkl')
    st.success("Model loaded successfully!")
except:
    st.error("Model file not found. Please run train_model.py first.")
    model = None



st.set_page_config(page_title="Parkinson's Disease Detection", page_icon="🎤")
st.title("🎤 Parkinson's Disease Risk Assessment from Voice")
st.write("This app records your voice and assesses the risk of Parkinson's disease using machine learning.")

# Instructions
st.subheader("📋 Instructions")
st.write("1. Click the 'Start Recording' button below")
st.write("2. Speak for 5 seconds (e.g., count from 1 to 10)")
st.write("3. Wait for the analysis results")
st.write("4. View your risk assessment")

# Recording function
def record_audio(duration=5, sample_rate=22050):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()
    return audio.flatten(), sample_rate

# Record button
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button('🎤 Start Recording', type='primary'):
        if model is None:
            st.error("Model not loaded. Please train the model first.")
        else:
            with st.spinner('Recording... Please speak now.'):
                audio, sr = record_audio()
                sf.write('temp_audio.wav', audio, sr)
            
            st.success("✅ Recording complete!")
            
            # Play the recorded audio
            st.audio('temp_audio.wav', format='audio/wav')
            
            # Extract features and predict
            with st.spinner('🔬 Analyzing voice features...'):
                time.sleep(2)  # Simulate processing time
                features = extract_features('temp_audio.wav')
                
                if features is not None:
                    # Make prediction
                    prediction = model.predict([features])
                    probability = model.predict_proba([features])
                    
                    risk_score = probability[0][1]  # Probability of Parkinson's
                    
                    # Display results
                    st.subheader("📊 Results")
                    
                    # Risk meter with color coding
                    if risk_score < 0.3:
                        st.success(f"✅ LOW RISK: Healthy with {(1 - risk_score)*100:.1f}% confidence")
                        st.balloons()
                    elif risk_score < 0.7:
                        st.warning(f"⚠️ MODERATE RISK: {(risk_score)*100:.1f}% probability of Parkinson's")
                    else:
                        st.error(f"🚨 HIGH RISK: Parkinson's detected with {(risk_score)*100:.1f}% probability")
                    
                    # Progress bar
                    st.progress(risk_score)
                    st.write(f"Risk Score: {risk_score:.3f} (0 = healthy, 1 = Parkinson's)")
                    
                    # Feature explanation
                    with st.expander("📈 View Extracted Features"):
                        feature_names = [
                            'Mean Pitch', 'Max Pitch', 'Min Pitch',
                            'Jitter Local', 'Jitter Absolute', 'Jitter RAP', 'Jitter PPQ5', 'Jitter DDP',
                            'Shimmer Local', 'Shimmer dB', 'Shimmer APQ3', 'Shimmer APQ5', 'Shimmer DDA',
                            'Harmonics-to-Noise Ratio'
                        ]
                        for name, value in zip(feature_names, features):
                            st.write(f"{name}: {value:.6f}")
                else:
                    st.error("❌ Error in feature extraction. Please try recording again.")
            
            # Clean up
            if os.path.exists('temp_audio.wav'):
                os.remove('temp_audio.wav')

# Disclaimer
st.markdown("---")
st.markdown("### ⚠️ Disclaimer")
st.markdown("This tool is for educational and research purposes only. It is not a medical diagnostic tool. Please consult healthcare professionals for medical advice.")

# Footer
st.markdown("---")
st.markdown("Built for CSA Hackathon | AI for Early Detection of Parkinson's from Voice")
st.markdown("By First Year Student")