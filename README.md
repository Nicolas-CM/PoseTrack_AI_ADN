# 🧠 PoseTrack AI – Exercise Classification System

## Final Project – Artificial Intelligence I

**Universidad Icesi – Department of Intelligent Computing and Systems**  
**Semester 2025-1**

---

## 🎯 Project Overview

**PoseTrack AI** is an intelligent tool for analyzing, classifying, and evaluating physical exercises in real time using pose estimation and machine learning models.  
It follows the **CRISP-DM** methodology and integrates tools like **MediaPipe** and datasets such as **MMFit** and **Kaggle Fitness Dataset**.

The system aims to contribute to **physiotherapy**, **personalized training**, and **injury prevention** by automatically recognizing movement patterns and providing posture analysis.

---

## 🎥 **Video of the Final Project Delivery**

Watch the full presentation and explanation of **PoseTrack AI**:

[https://youtu.be/cDP3GSQNX2k?si=d7lbIf0mUNeByopG](https://youtu.be/cDP3GSQNX2k?si=d7lbIf0mUNeByopG)

---

## 🎥 **Real-time Exercise Classification Demo**

See **PoseTrack AI** in action with a live demo:

[https://youtu.be/qUk4IYl5pug?si=LfcLILjKWC4wcJ9T](https://youtu.be/qUk4IYl5pug?si=LfcLILjKWC4wcJ9T)

---



## 📦 How to Run

### ▶️ 1. Navigate to the Project Folder

From the **root of the repository**, run:

```bash
cd Entrega3
````

### 🧠 2. Install Requirements

Make sure you have Python 3.8+ and install dependencies:

```bash
pip install -r requirements.txt
```

### 🏋️‍♂️ 3. Train the Models

To extract features and train the classification models:

```bash
python main.py train
```

This will:

* Process training videos
* Extract pose-based features
* Train and save both models in the `/models/` folder

### 🖥 4. Launch the Interface

To run the real-time GUI:

```bash
python main.py gui
```

Once the GUI opens:

1. Click **"Start Recording"** to activate the webcam
2. Select a **model** (*Basic* or *Gym*)
3. Perform the **physical activity** in front of the camera
4. Watch **predictions and metrics in real time**

---

## 📂 Project Structure

```
Entrega3/
├── data/               # Training videos and annotations
├── models/             # Trained ML model files
├── features/           # Extracted features (.pkl)
├── gui/                # GUI components
├── main.py             # Entry point
├── utils/              # Helper functions
└── requirements.txt    # Python dependencies
```

---

## 🧠 Key Technologies

* **Pose Estimation:** MediaPipe
* **Feature Engineering:** Joint angles, velocities, trunk inclination
* **Machine Learning Models:** XGBoost, Random Forest, SVM
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score
* **Interface:** Tkinter-based GUI with webcam support

---

## 📊 Datasets Used

* 🧘 **MMFit Dataset:**
  [Official Site](https://mmfit.github.io) | [GitHub](https://github.com/KDMStromback/mm-fit)

* 🏋️ **Kaggle Workout/Fitness Dataset:**
  [Kaggle Link](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)

* 🎥 **Custom Videos:**
  Recorded by team members for additional activities

---

## 👥 Authors

* [Davide Flamini](https://github.com/davidone007)
* [Andrés Cabezas](https://github.com/andrescabezas26)
* [Nicolás Cuellar](https://github.com/Nicolas-CM)
