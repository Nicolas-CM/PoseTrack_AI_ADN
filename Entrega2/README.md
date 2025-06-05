# 🎯 PoseTrack AI - Delivery 2: Model Training and Evaluation

## 📖 Overview

This document presents the second delivery of the PoseTrack AI project, which focuses on the implementation of data acquisition strategies, data preparation, machine learning model training (including hyperparameter tuning), evaluation results, and deployment planning. Additionally, we provide an initial analysis of the solution's impact in the context where the problem is addressed.

## 🎯 Project Scope

### Current Implementation (Delivery 2)
The current implementation focuses on **basic human movement classification** using pose estimation techniques. We have successfully developed and trained models to recognize six fundamental activities:

- **Approaching** (acercarse): Moving towards the camera
- **Moving away** (alejarse): Moving away from the camera  
- **Turn right** (girarD): Rotating to the right
- **Turn left** (girarI): Rotating to the left
- **Sitting down** (sentarse): Sitting motion
- **Standing up** (levantarse): Standing up motion

### Future Implementation (Final Delivery)
For the final delivery, we will expand the system to include **gym exercise recognition** with three additional exercises:

- **Push-ups**: Upper body strength exercise
- **Squats**: Lower body strength exercise  
- **Barbell biceps curl**: Isolated arm exercise

These exercise videos have been sourced from the [Workout/Fitness Video Dataset](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video) on Kaggle, which provides high-quality training data for fitness activity recognition.

## 📊 Data Acquisition Strategy

### 📹 Video Data Collection
Our data acquisition strategy combines two complementary approaches:

#### 1. Self-Recorded Videos
- **Custom dataset**: We recorded our own videos to ensure data quality and consistency
- **Controlled environment**: Videos were captured in standardized conditions
- **Multiple subjects**: Data from different individuals to improve model generalization
- **Varied perspectives**: Different camera angles and distances

#### 2. External Dataset Integration
- **Source**: [Kaggle Workout/Fitness Video Dataset](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)
- **Purpose**: Gym exercise recognition for final delivery
- **Quality**: Professional fitness videos with clear exercise demonstrations
- **Diversity**: Multiple subjects performing standardized exercises

### 🔧 Feature Extraction Process
1. **Pose Detection**: MediaPipe framework for keypoint extraction
2. **Feature Engineering**: Joint angles, distances, and temporal patterns
3. **Data Normalization**: Standardization of pose coordinates
4. **Temporal Windowing**: Frame-based feature aggregation

## 📈 Data Analysis and Preparation

### 📊 Dataset Statistics
Our current dataset contains **6,393 samples** with the following class distribution:

| Activity | Samples | Percentage |
|----------|---------|------------|
| Sitting down (sentarse) | 1,579 | 24.7% |
| Standing up (levantarse) | 1,240 | 19.4% |
| Turn right (girarD) | 1,149 | 18.0% |
| Approaching (acercarse) | 919 | 14.4% |
| Moving away (alejarse) | 892 | 14.0% |
| Turn left (girarI) | 614 | 9.6% |

**Total features per sample**: 281 pose-based characteristics

### 🔄 Data Preprocessing
- **Missing value handling**: Interpolation for incomplete pose detections
- **Feature scaling**: Standardization using scikit-learn StandardScaler
- **Class balancing**: Analysis of distribution for potential augmentation needs
- **Train-test split**: 80/20 stratified split for evaluation

## 🤖 Model Training and Hyperparameter Tuning

### 🧠 Implemented Models

#### 1. Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Hyperparameter tuning**: Grid search for C and gamma parameters
- **Cross-validation**: 5-fold stratified CV

#### 2. Random Forest (RF)
- **Trees**: Optimized number of estimators
- **Features**: sqrt(n_features) per split
- **Hyperparameter tuning**: Grid search for max_depth and n_estimators

#### 3. XGBoost (XGB)
- **Boosting**: Gradient boosting framework
- **Regularization**: L1 and L2 regularization
- **Hyperparameter tuning**: Bayesian optimization for learning rate, max_depth, and subsample

### ⚙️ Training Configuration
- **Cross-validation**: 5-fold stratified
- **Evaluation metrics**: Precision, Recall, F1-score, Accuracy
- **Model persistence**: Joblib serialization for deployment

## 📊 Results and Performance Evaluation

### 🏆 Model Performance Comparison

| Model | Test Accuracy | CV Mean | CV Std | Status |
|-------|---------------|---------|--------|--------|
| **Random Forest** | **100.00%** | **100.00%** | **0.0000** | 🥇 **Best** |
| **XGBoost** | **100.00%** | **99.94%** | **0.0008** | 🥈 Second |
| **SVM** | **90.77%** | **91.67%** | **0.0045** | 🥉 Third |

### 📈 Detailed Performance Analysis

#### 🌟 Random Forest (Best Model)
```
Classification Report:
              precision    recall  f1-score   support
   acercarse       1.00      1.00      1.00       184
    alejarse       1.00      1.00      1.00       178
      girarD       1.00      1.00      1.00       230
      girarI       1.00      1.00      1.00       123
  levantarse       1.00      1.00      1.00       248
    sentarse       1.00      1.00      1.00       316

    accuracy                           1.00      1279
   macro avg       1.00      1.00      1.00      1279
weighted avg       1.00      1.00      1.00      1279
```

#### ⚡ XGBoost
```
Classification Report:
              precision    recall  f1-score   support
   acercarse       1.00      1.00      1.00       184
    alejarse       1.00      1.00      1.00       178
      girarD       1.00      1.00      1.00       230
      girarI       1.00      1.00      1.00       123
  levantarse       1.00      1.00      1.00       248
    sentarse       1.00      1.00      1.00       316

    accuracy                           1.00      1279
   macro avg       1.00      1.00      1.00      1279
weighted avg       1.00      1.00      1.00      1279
```

#### 🔧 SVM
```
Classification Report:
              precision    recall  f1-score   support
   acercarse       1.00      1.00      1.00       184
    alejarse       1.00      1.00      1.00       178
      girarD       0.89      1.00      0.94       230
      girarI       1.00      0.80      0.89       123
  levantarse       0.96      0.65      0.78       248
    sentarse       0.79      0.98      0.87       316

    accuracy                           0.91      1279
   macro avg       0.94      0.90      0.91      1279
weighted avg       0.92      0.91      0.90      1279
```

### 📊 Key Insights
- **Random Forest** achieved perfect performance, demonstrating excellent capability for pose-based activity classification
- **XGBoost** showed near-perfect performance with high stability
- **SVM** performed well but showed some challenges with specific activities (levantarse, sentarse)
- The high performance across models indicates good feature engineering and data quality

## 🚀 Deployment Plan

### 🏗️ Architecture Overview
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Pose Detection  │───▶│ Feature Extract │
│   (Camera/File) │    │   (MediaPipe)    │    │   (281 features)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Activity Class │◀───│  ML Prediction   │◀───│ Data Processing │
│   + Confidence  │    │ (Random Forest)  │    │  (Normalization)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🔧 Technical Implementation
1. **Real-time Processing**: OpenCV for video capture and processing
2. **Pose Estimation**: MediaPipe for keypoint detection
3. **Model Inference**: Joblib-loaded Random Forest model
4. **User Interface**: GUI application for real-time feedback
5. **Data Pipeline**: Automated feature extraction and normalization

### 📱 Deployment Targets
- **Desktop Application**: Cross-platform GUI application
- **Web Application**: Browser-based interface for accessibility
- **Mobile Integration**: Future mobile app development
- **API Service**: RESTful API for third-party integration

## 🌍 Impact Analysis

### 🏥 Healthcare and Physiotherapy Impact

#### Positive Impacts
- **Objective Assessment**: Quantitative movement analysis reduces subjective bias
- **Remote Monitoring**: Enables therapy sessions outside clinical settings
- **Progress Tracking**: Longitudinal analysis of patient improvement
- **Cost Reduction**: Decreases need for constant professional supervision
- **Accessibility**: Makes movement analysis available to underserved populations

#### Potential Challenges
- **Technology Adoption**: Healthcare professionals may require training
- **Data Privacy**: Patient movement data requires secure handling
- **Clinical Validation**: Need for extensive clinical trials and validation
- **Equipment Requirements**: Patients need access to cameras and computers

### 🏋️ Fitness and Training Impact

#### Positive Impacts
- **Form Correction**: Real-time feedback on exercise technique
- **Injury Prevention**: Early detection of improper movement patterns
- **Personalized Training**: Adaptation based on individual movement capabilities
- **Motivation Enhancement**: Gamification through accurate movement tracking

#### Considerations
- **User Engagement**: System must be intuitive and motivating
- **Accuracy Requirements**: High precision needed for safety-critical applications
- **Integration**: Compatibility with existing fitness platforms and devices

### 🔒 Ethical and Privacy Considerations
- **Data Consent**: Clear consent mechanisms for video and pose data collection
- **Data Minimization**: Only collecting necessary pose information
- **Anonymization**: Removing identifying features from stored data
- **Transparency**: Clear communication about how the system works and its limitations

## 🛣️ Future Development Roadmap

### 📅 Phase 1: Final Delivery (Current Focus)
- **Gym Exercise Integration**: Implement push-ups, squats, and biceps curls recognition
- **Enhanced GUI**: Improved user interface with exercise selection
- **Performance Optimization**: Model optimization for real-time processing
- **Documentation**: Complete user manuals and technical documentation

### 📅 Phase 2: Advanced Features
- **Exercise Counting**: Automatic repetition counting for gym exercises
- **Form Analysis**: Detailed technique assessment with corrective feedback
- **Progress Tracking**: Historical performance analysis and trends
- **Multi-person Detection**: Support for multiple individuals simultaneously

### 📅 Phase 3: Clinical Integration
- **Medical Validation**: Clinical studies and validation protocols
- **Healthcare Integration**: EMR system integration and clinical workflows
- **Regulatory Compliance**: HIPAA compliance and medical device certification
- **Professional Training**: Healthcare provider education and certification programs

## 📁 Project Structure

```
Entrega2/
├── 📄 README.md                    # This documentation
├── 📄 execution-guide.md           # Technical execution guide
├── 🐍 main.py                      # Main training script
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package configuration
├── ⚙️ config/
│   └── settings.py                 # Configuration parameters
├── 📊 data/
│   ├── training_data_*.pkl         # Processed training datasets
├── 🤖 models/
│   ├── rf_model_*.pkl              # Random Forest models
│   ├── svm_model_*.pkl             # SVM models
│   └── xgb_model_*.pkl             # XGBoost models
├── 🔧 src/
│   ├── core/                       # Core functionality
│   ├── gui/                        # User interface
│   ├── training/                   # Model training modules
│   └── utils/                      # Utility functions
└── 🎬 Videos/                      # Training video dataset
    ├── acercarse-*.mp4             # Approaching videos
    ├── alejarse-*.mp4              # Moving away videos
    ├── girarD-*.mp4                # Turn right videos
    ├── girarI-*.mp4                # Turn left videos
    ├── levantarse-*.mp4            # Standing up videos
    └── sentarse-*.mp4              # Sitting down videos
```

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run training**: `python main.py`
4. **Launch application**: Follow execution guide for GUI usage

## 👥 Development Team

This project is developed as part of an AI course final project, focusing on practical applications of machine learning in healthcare and fitness domains.

---

*For detailed technical implementation and usage instructions, please refer to the [execution-guide.md](execution-guide.md) file.*
