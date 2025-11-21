# 🖐️ Sign-to-Text-Language  
### *Real-time Sign Language Recognition → Text → Speech using MediaPipe + CNN + Flask*

A complete end–to–end pipeline that detects hand landmarks from a webcam using **MediaPipe**, extracts **63 keypoint coordinates**, classifies them using a **trained Conv1D neural network**, builds words/sentences, gives **word suggestions**, and converts the detected text to **speech** using **gTTS**.

This repository contains the full source code, dataset creation notebook, trained model, and a Flask-based web application.

---

## 📌 Table of Contents
- 🌟 Features
- 📂 Repository Structure
- 🧠 System Architecture
- ⚙️ Installation & Setup
- ▶️ Running the Application
- 🧩 Model Details
- 🖐️ How Gesture Detection Works
- 🔤 Word Building & Suggestions
- 🌍 Translation & Text-to-Speech
- 🛠️ API Endpoints
- 📘 Data Creation & Training Notebook
- 🚀 Future Improvements
- 🤝 Contributing
- 📄 License

---

# 🌟 Features

### ✔ Real-time Hand Landmark Detection  
Uses **MediaPipe Hands** to detect **21 hand landmarks** per frame.

### ✔ Deep Learning Model for Sign Classification  
Predicts **A–Z**, plus **SPACE** and **SUBMIT** using the included CNN model.

### ✔ Flask-based Web Interface  
Live video stream + interactive buttons for registering letters & sentences.

### ✔ Word Suggestion Engine  
Uses **NLTK word corpus** to suggest context-based English words.

### ✔ Sentence Building  
Detected letters → words → full sentences with front-end display.

### ✔ Multilingual Translation  
Translates text using `googletrans`.

### ✔ Text-to-Speech Support  
Converts text into speech using **gTTS** (supports many languages).

### ✔ Training Notebook Included  
Full dataset creation + training workflow included in Jupyter Notebook.

---

# 📂 Repository Structure

```
Sign-to-Text-Language/
├── app1.py
├── win_app.py
├── sign_language_model.h5
├── label_classes.npy
├── Data Creation & Model Analysis.ipynb
│
├── templates/
│   ├── index.html
│   └── speak.html
│
├── sign_data/
│
├── requirements.txt
├── environment.yml
├── win_env.yml
└── temp_audio.mp3
```

---

# 🧠 System Architecture

### 🎥 1. Frame Capture  
Flask streams webcam frames using a generator.

### 🖐️ 2. Landmark Extraction  
MediaPipe → 21 hand points → **63 values**.

### 🧮 3. CNN Model Prediction  
Classifies gestures into 27 classes.

### 📝 4. Word / Sentence Building  

### 💡 5. Word Suggestions  
Uses NLTK `words` list.

### 🌍 6. Translation & Speech  
Using googletrans + gTTS.

---

# ⚙️ Installation & Setup

```
git clone https://github.com/Atharva0177/Sign-to-Text-Language.git
cd Sign-to-Text-Language
```

### Virtual Environment

```
python -m venv venv
venv\Scriptsctivate
```

### Install Dependencies

```
pip install -r requirements.txt
```

---

# ▶️ Running the Application

```
python app1.py
```

Open:

```
http://127.0.0.1:5000/
```

---

# 🧩 Model Details

- Conv1D architecture  
- Input: 63 features  
- Output: 27 classes  
- Trained using included notebook  

---

# 🖐️ How Gesture Detection Works

- ROI-based detection  
- 21 landmarks extracted  
- Flattened into (63,)  
- Passed to CNN model  

---

# 🔤 Word Building & Suggestions

- Live prediction  
- Register letter  
- Build words/sentences  
- NLTK suggestion engine  

---

# 🌍 Translation & Text-to-Speech

- `/speak` endpoint  
- googletrans for translation  
- gTTS for speech (MP3 output)

---

# 🛠️ API Endpoints

- `/` — UI  
- `/video_feed` — stream  
- `/register_letter` — classify & append  
- `/suggest` — word suggestions  
- `/clear_detections`  
- `/clear_all`  
- `/speak` — audio output  

---

# 📘 Data Creation & Training Notebook

Contains:
- Data capture  
- Preprocessing  
- Model training  
- Accuracy curves  

---

# 🚀 Future Improvements

- Temporal smoothing  
- Two-hand support  
- More gesture classes  
- ONNX export  
- Offline TTS  

---


