PowerShell run steps for Parkinson Voice project

1. Open PowerShell and change into the project folder:

```powershell
Set-Location -Path "d:\hacksphere\parkinson_voice_detection"
```

2. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Download dataset (if missing):

```powershell
if (-Not (Test-Path -Path .\sample_data)) { New-Item -ItemType Directory -Path .\sample_data }
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data" -OutFile ".\sample_data\parkinsons.data"
```

4. Run training script (example):

```powershell
.\.venv\Scripts\python.exe .\train_model.py --data .\sample_data\parkinsons.data --output .\parkinson_model.pkl
```

5. Run the Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```
