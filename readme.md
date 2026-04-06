
# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using **YOLOv8** + **MediaPipe Face Mesh**.  
It detects closed eyes using **Eye Aspect Ratio (EAR)** and yawning using **Mouth Aspect Ratio (MAR)**, then triggers an alarm if drowsiness persists.

---

## 🚀 Features
- YOLOv8 for face/eye/yawn detection
- MediaPipe for accurate facial landmarks
- EAR + MAR calculation with smoothing
- Alarm sound when drowsy for > 4 seconds
- Real-time performance on webcam

---

## 🛠 Tech Stack
- Python
- Ultralytics YOLOv8
- MediaPipe
- OpenCV
- Pygame (for alarm)


---

## ⚙️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

2. **Create virtual environment**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1     # For PowerShell (Windows)
# or source venv/bin/activate   # For Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the detection**
```bash
cd src
python detect_drowsiness.py
```

---

## 📊 Dataset
Trained on Roboflow Dataset

---

## 📈 Results
- Normal open eyes: EAR ≈ 0.29  
- Drowsy (eyes closed): EAR drops significantly  
- Real-time FPS: ~25–35 on CPU  

---

## 🔮 Future Improvements
- Head pose estimation
- Calibration mode for different users
- Mobile deployment (ONNX/TFLite)

---



