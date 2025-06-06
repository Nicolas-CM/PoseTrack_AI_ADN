# 📦 Deliverable 1: Intelligent System for Exercise Classification and Evaluation

> A project focused on classifying physical exercises and evaluating their correctness in real time using pose estimation and joint analysis.

## 📌 Project Repository
🔗 [GitHub Repository](https://github.com/Nicolas-CM/PoseTrack_AI_ADN.git)

## 🎯 General Objective
To develop an intelligent system capable of:
- Classifying physical exercises (Push-ups, Barbell Bicep Curls, Squats).
- Evaluating whether the exercises are performed correctly using real-time pose analysis.
- Providing real-time feedback on posture and movement quality.

---

## ❓ Main Research Question

**How can we develop an intelligent system that classifies and evaluates physical exercises in real time using pose estimation?**

### Subquestions:
- Which joint-based metrics (e.g., angles, trunk inclination) are most relevant for evaluating correctness?
- How can the system provide real-time feedback during exercises?
- How can this be applied in physiotherapy and personalized fitness?

---

## 🧠 Problem Type

This is a **supervised classification** problem in the field of artificial intelligence and computer vision, focusing on:
- Human Activity Recognition (HAR)
- Pose-based movement assessment
- Multimodal signal processing (video + sensors)

---

## 📐 Methodology: CRISP-DM

### 1. Business Understanding
- Purpose: Assess and correct exercise performance in real time.
- Applications: Personalized training, rehabilitation, physiotherapy at home.

### 2. Data Understanding
- Source: [Workout/Exercises Video Dataset – Kaggle](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)
- Videos grouped by exercise type in folder structure.

### 3. Data Preparation
- Pose extraction using **MediaPipe** (2D).
- Feature engineering:
  - Joint angles
  - Trunk inclination
  - Joint velocity
- Data normalization and segmentation.

### 4. Modeling
- Classifiers: SVM, Random Forest, XGBoost.
- Model selection based on performance, with cross-validation and hyperparameter tuning.

### 5. Evaluation
- Standard metrics: Accuracy, Recall, F1-Score.
- Pose-specific metrics:
  - Joint angle error
  - Trunk alignment deviation
  - Symmetry between left and right limbs

### 6. Deployment
- Development of a **real-time graphical interface** with:
  - Webcam-based video feed
  - Activity detection
  - Pose visualization and corrective feedback

---

## 📁 Dataset Overview

- **Exercise types considered**: Squats, Barbell Bicep Curls, Push-ups (three selected for the final version).
- **Modality**: Folder names = Activity labels
- **Limitations**:
  - No wearable sensor data
  - Labels inferred from folder structure and video content

---

## 📊 Evaluation Metrics

- **Accuracy**: Correct classifications over total instances.
- **Recall**: Ability to detect correct movement execution.
- **F1-Score**: Balance between precision and recall.
- **Inference Time**: Efficiency of real-time detection.

### Pose-Specific Metrics:
- **Joint Angle Error**: Difference from ideal angle (e.g., 90° for squats).
- **Trunk Inclination**: Misalignment between head, shoulders, and hips.
- **Symmetry**: Movement consistency between left/right limbs.

---

## 📈 Data Expansion Strategies

If additional data is needed, the team may:
- Record new videos at different angles, speeds, and environments.
- Generate synthetic pose data via simulation tools.
- Use annotation platforms (LabelStudio, CVAT) for new data.
- Collaborate with physiotherapists or fitness professionals.

---

## ⚖️ Ethical Considerations

- **Informed Consent**: Required from all participants being recorded.
- **Data Privacy**: All video data is anonymized and securely stored.
- **Inclusivity**: The system should work across diverse ages, body types, and physical abilities.

---

## 🔄 Next Steps

- Complete exploratory data analysis.
- Train and validate initial models.
- Build the real-time interface.
- Iterate based on testing and performance.
- Finalize selection of 3 target exercises for deployment.

---

