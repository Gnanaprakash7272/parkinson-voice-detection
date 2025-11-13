# 🎤 Parkinson's Voice Detection - AI Hackathon Project

A machine learning web application that analyzes voice recordings to assess the risk of Parkinson's disease.

## 📋 Features

- 🎙️ **Real-time Voice Recording** – Record 5-second audio samples
- 🧠 **AI-Powered Analysis** – Extract 14 voice biomarkers using praat-parselmouth
- 📊 **Risk Assessment** – ML model predicts Parkinson's probability
- 🎨 **Beautiful UI** – Streamlit-based web interface with visual feedback
- 📈 **Feature Breakdown** – View all extracted voice features

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd hacksphere
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r parkinson_voice_detection/requirements.txt
   ```

4. **Train the model** (if `parkinson_model.pkl` doesn't exist)
   ```bash
   python parkinson_voice_detection/train_model.py
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run parkinson_voice_detection/app.py
   ```

The app will be available at `http://localhost:8501`

## 📁 Project Structure

```
hacksphere/
├── parkinson_voice_detection/
│   ├── app.py                    # Main Streamlit app
│   ├── extract_features.py       # Voice feature extraction
│   ├── train_model.py            # Model training script
│   ├── requirements.txt          # Python dependencies
│   ├── parkinson_model.pkl       # Trained ML model
│   └── sample_data/              # Sample audio files
├── demo.py                       # Demo script
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## 🔧 How It Works

1. **Audio Recording** – User records 5 seconds of voice
2. **Feature Extraction** – Voice features extracted using Praat algorithms:
   - Pitch (F0), jitter, shimmer, HNR
   - 14 biomarkers total
3. **ML Prediction** – Scikit-learn model predicts probability
4. **Risk Display** – Color-coded result (Low/Moderate/High)

## 📚 Technologies Used

- **Frontend**: Streamlit 1.28.0
- **Audio Processing**: librosa, soundfile, praat-parselmouth
- **ML**: scikit-learn 1.3.0
- **Data**: NumPy, Pandas
- **Visualization**: Plotly, Matplotlib

## ⚠️ Disclaimer

This is an **educational and research tool**, not a medical diagnostic application.  
Always consult a qualified healthcare professional for medical advice.



## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Pull requests welcome! Please fork the repository and create a branch for your changes.

---

**Have questions?** Open an issue or reach out to the project maintainers.
