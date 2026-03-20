# 🏋️ Powerlifting AI Analyzer

> Computer vision system that automatically analyzes powerlifting technique from video using MediaPipe and Python.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat&logo=OpenCV&logoColor=white)
![Status](https://img.shields.io/badge/Status-Functional-brightgreen?style=flat)

---

## 📌 Overview

This project uses **computer vision** and **pose estimation** to analyze the three main powerlifting exercises from recorded video:

- 🟢 **Squat** (Sentadilla)
- 🟢 **Bench Press** (Press de banca)
- 🟢 **Deadlift** (Peso muerto)

The system detects 33 full-body landmarks using **MediaPipe Pose**, calculates joint angles in real time, counts repetitions automatically, and provides structured feedback on movement quality — helping coaches and athletes identify form errors.

---

## 🎬 Demo

> 📸 *Screenshots and video demo coming soon — add your images to the `/docs` folder*



---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Pose Estimation | MediaPipe Pose |
| Video Processing | OpenCV |
| Analysis | Custom joint angle & rep counting algorithms |
| Interface | Python GUI / Web interface |

---

## ⚙️ Features

- ✅ Detects and classifies 3 powerlifting movements automatically
- ✅ Full-body 33-point landmark detection per frame
- ✅ Real-time joint angle calculation (knees, hips, shoulders, elbows)
- ✅ Automatic repetition counter
- ✅ Movement quality feedback
- ✅ Video file input with frame-by-frame analysis
- ✅ Visual overlay of skeleton and angles on video

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/david323902/Proyecto-de-powerlifter.git
cd Proyecto-de-powerlifter

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the analyzer with a video file
python main.py --video path/to/your/video.mp4

# Or launch the interface
python app.py
```

---

## 📁 Project Structure

```
Proyecto-de-powerlifter/
├── main.py              # Entry point
├── app.py               # Interface launcher
├── analyzer/
│   ├── pose_detector.py # MediaPipe integration
│   ├── angle_calc.py    # Joint angle calculation
│   └── rep_counter.py   # Repetition counting logic
├── models/              # Exercise classification
├── docs/                # Screenshots and demo media
└── requirements.txt
```

---

## 👤 Author

**Johan David Toro Ortiz** — sole developer (engineering + implementation)  
Documentation by: teammate collaboration  
📧 davidortiz634@gmail.com · [LinkedIn](https://www.linkedin.com/in/johan-david-toro-ortiz-512680349/) · [GitHub](https://github.com/david323902)

---

## 🇪🇸 Descripción en español

Sistema de visión por computadora que analiza la técnica en los tres ejercicios principales del powerlifting (sentadilla, press de banca y peso muerto) a partir de video grabado. Utiliza **MediaPipe** para la detección de 33 puntos corporales, calcula ángulos articulares, cuenta repeticiones automáticamente y genera retroalimentación sobre la calidad del movimiento. Proyecto desarrollado íntegramente por Johan David Toro Ortiz como parte de su formación en ingeniería de sistemas.
