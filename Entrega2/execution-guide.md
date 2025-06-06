# 🎯 PoseTrack AI - Execution and Usage Guide

## 📋 General Description

PoseTrack AI is an intelligent real-time movement analysis system that uses pose estimation techniques and machine learning models to classify and evaluate physical activities. The system is designed to contribute to the fields of physiotherapy and functional training.

## 🎯 Main Features

### 🔍 Real-Time Analysis
- **Pose detection**: Uses MediaPipe to extract human body keypoints
- **Activity classification**: Identifies 6 types of activities:
  - 🚶‍♂️ **Approaching**: Walking towards the camera
  - 🚶‍♀️ **Moving away**: Walking away from the camera
  - ↩️ **Turn right**: Rotation movement to the right
  - ↪️ **Turn left**: Rotation movement to the left
  - 🪑 **Sitting down**: Sitting action
  - 🧍 **Standing up**: Standing up action

### 📊 Postural Metrics
- **Joint angles**: Elbows, knees, hips and trunk inclination
- **Detection confidence**: Accuracy of pose detection
- **Temporal analysis**: Movement tracking over time

### 🤖 Machine Learning Models
- **SVM (Support Vector Machine)**: Classification with RBF kernel
- **Random Forest**: Decision tree ensemble
- **XGBoost**: Optimized gradient boosting

## 🛠️ System Requirements

### 📋 Minimum Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Camera**: Webcam or USB camera
- **RAM**: Minimum 4GB (recommended 8GB)
- **Processor**: Multi-core CPU (GPU optional)

### 📦 Dependencies
```txt
opencv-python>=4.8.0
mediapipe>=0.10.13
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
joblib>=1.3.0
Pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
scipy>=1.11.0
```

## 🚀 Installation and Configuration

### 1️⃣ Environment Setup

```bash
# Navigate to the project directory
cd "PoseTrack_AI_ADN/Entrega 2"

# Check Python version
python --version

# Run setup script
python setup.py
```

### 2️⃣ Manual Dependency Installation

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt

# Or install individual dependencies
pip install opencv-python mediapipe numpy pandas scikit-learn xgboost joblib pillow matplotlib seaborn tqdm scipy
```

### 3️⃣ Installation Verification

```bash
# Verify that all dependencies are installed
python -c "import cv2, mediapipe, numpy, pandas, sklearn, xgboost, joblib; print('✅ All dependencies are installed')"
```

## 🎮 Execution Options

### 🖥️ Graphical Interface (Main Mode)

```bash
# Start with graphical interface (default mode)
python main.py

# Or explicitly
python main.py gui
```

**Interface Features:**
- 📹 **Camera control**: Start/stop video capture
- 🎯 **Model selector**: Load pre-trained models
- 📊 **Metrics panel**: Real-time visualization of:
  - Detected activity and confidence
  - Joint angles
  - System FPS
  - System status log
- ⚙️ **Configuration**: Adjust camera and detection parameters

### 🏋️ Training Mode

```bash
# Train new models
python main.py train
```

**Training process:**
1. **Feature extraction** from videos in the `Videos/` folder
2. **Model training** SVM, Random Forest and XGBoost
3. **Evaluation and saving** of models with metrics
4. **Automatic comparison** to identify the best model

### ℹ️ Help and Commands

```bash
# Show help
python main.py --help
python main.py -h
```

## 🎛️ System Configuration

### 📹 Camera Configuration
- **Resolution**: 640x480, 800x600, 1280x720
- **FPS**: 15-60 (recommended: 30)
- **Auto-detection**: The system detects available cameras

### 🔧 MediaPipe Configuration
- **Model complexity**: 0 (light), 1 (medium), 2 (full)
- **Minimum detection confidence**: 0.1 - 1.0 (default: 0.5)
- **Tracking confidence**: 0.1 - 1.0 (default: 0.5)

### 🧠 Classification Configuration
- **Analysis window**: 10-60 frames (default: 30)
- **Smoothing buffer**: 5-20 predictions (default: 10)
- **Feature normalization**: Enabled by default

## 📁 Project Structure

```
PoseTrack_AI_ADN/
├── 📄 README.md
├── 📁 Entrega2/
│   ├── 🐍 main.py                 # Main entry point
│   ├── 🐍 setup.py               # Setup script
│   ├── 📋 requirements.txt       # Project dependencies
│   ├── 📁 config/
│   │   └── 🐍 settings.py        # Global configuration
│   ├── 📁 src/
│   │   ├── 📁 core/               # Core modules
│   │   │   ├── 🐍 pose_tracker.py        # Pose detection
│   │   │   ├── 🐍 feature_extractor.py   # Feature extraction
│   │   │   └── 🐍 activity_classifier.py # Activity classification
│   │   ├── 📁 gui/                # Graphical interface
│   │   │   └── 🐍 main_gui.py     # Main interface
│   │   ├── 📁 training/           # Training system
│   │   │   └── 🐍 train_model.py  # Model training
│   │   └── 📁 utils/              # Utilities
│   ├── 📁 models/                 # Trained models
│   ├── 📁 data/                   # Processed training data
│   └── 📁 Videos/                 # Videos for training
```

## 🎬 Using the Graphical Interface

### 🚀 Quick Start

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **Select camera**: 
   - Use the "Camera" dropdown to select device
   - Use the 🔄 button to refresh camera list

3. **Load model**:
   - Select model from "Current Model" dropdown
   - Or use "📁 Load Model" for custom files

4. **Start analysis**:
   - Click "▶️ Start Camera"
   - Observe real-time metrics

### 🎯 Control Panel

#### 🎮 Main Buttons
- **▶️ Start Camera / ⏹️ Stop Camera**: Video capture control
- **📁 Load Model**: Load custom model from file
- **🔄 Train Model**: Open training dialog
- **⚙️ Settings**: Access system settings

- **Detected Activity**: Shows current activity and confidence level
- **Postural Metrics**: Real-time joint angles:
  - Left and right elbows
  - Left and right knees
  - Left and right hips
  - Trunk inclination
- **System Status**: Event log and application state

### 🏋️ Model Training

1. **Access**: Click "🔄 Train Model"

2. **Model selection**:
   - ✅ SVM: Support Vector Machine
   - ✅ Random Forest: Random forest
   - ✅ XGBoost: Gradient boosting

3. **Automatic process**:
   - Feature extraction from videos
   - Training and evaluation
   - Automatic saving with timestamp
   - Metrics reporting

## 📈 Training Data

### 📹 Training Videos

The system uses videos located in the `Videos/` folder with the following nomenclature:

```
acercarse-[person]-[number].[extension]
alejarse-[person]-[number].[extension]
girarD-[person]-[number].[extension]
girarI-[person]-[number].[extension]
sentarse-[person]-[number].[extension]
levantarse-[person]-[number].[extension]
```

**Examples**:
- `acercarse-davide-1.MOV`
- `sentarse-cabezas-2.mp4`
- `girarD-nicolas-1.MOV`

### 🔄 Automatic Processing

1. **Activity detection**: Based on filename
2. **Feature extraction**: 
   - MediaPipe landmarks (33 body points)
   - Calculated joint angles
   - Temporal features (velocity, acceleration)
   - Trajectory features
3. **Sliding window**: 30 frames per sample (configurable)

## 🤖 Available Models

### 🔍 SVM (Support Vector Machine)
- **Parameters**: C=1.0, kernel='rbf', gamma='scale'
- **Features**: Excellent for non-linear classification
- **Use**: Analysis of complex movement patterns

### 🌳 Random Forest
- **Parameters**: 100 estimators, max_depth=10
- **Features**: Robust against overfitting
- **Use**: Feature interpretability

### 🚀 XGBoost
- **Parameters**: 100 estimators, depth=6, learning_rate=0.1
- **Features**: High accuracy and efficiency
- **Use**: Maximum predictive performance

## 📊 Metrics and Evaluation

### 🎯 Training Metrics
- **Accuracy**: Percentage of correct predictions
- **Cross-validation**: 5-fold cross-validation
- **Confusion matrix**: Detailed analysis by class
- **Classification report**: Precision, recall, F1-score

### 📈 Real-Time Metrics
- **Prediction confidence**: 0-100%
- **System FPS**: Frames per second processed
- **Pose confidence**: Quality of landmark detection
- **Temporal smoothing**: Prediction buffer for stability

## 🔧 Troubleshooting

### ❌ Common Problems

#### 🎥 Camera not detected
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"
```

#### 📦 Dependency errors
```bash
# Reinstall dependencies
pip uninstall opencv-python
pip install opencv-python

# For macOS with M1/M2
pip install opencv-python --no-cache-dir
```

#### 🤖 Model won't load
1. Verify that the `.pkl` file exists in `models/`
2. Check version compatibility
3. Review error log in the interface

#### 🐌 Slow performance
1. Reduce camera resolution to 640x480
2. Decrease FPS to 15-20
3. Use MediaPipe model complexity = 0
4. Close other applications using the camera

### 🆘 Diagnostic Commands

```bash
# Verify complete installation
python setup.py

# Test individual components
python -c "from src.core.pose_tracker import PoseTracker; print('✅ PoseTracker OK')"
python -c "from src.core.activity_classifier import ActivityClassifier; print('✅ ActivityClassifier OK')"

# Verify MediaPipe
python -c "import mediapipe as mp; print('✅ MediaPipe version:', mp.__version__)"
```

## 📝 Logs and Generated Files

### 📁 Generated Files Structure

```
models/
├── svm_model_20250605_172734.pkl      # Trained SVM model
├── svm_model_20250605_172734.json     # Model metrics
├── rf_model_20250605_172810.pkl       # Random Forest model
├── rf_model_20250605_172810.json      # Model metrics
└── xgb_model_20250605_172834.pkl      # XGBoost model

data/
├── training_data_20250605_172734.pkl  # Processed training data
└── feature_cache/                     # Extracted features cache
```

### 📋 Naming Format
- **Timestamp**: YYYYMMDD_HHMMSS
- **Models**: `{type}_model_{timestamp}.pkl`
- **Metrics**: `{type}_model_{timestamp}.json`
- **Data**: `training_data_{timestamp}.pkl`

## 🔮 Advanced Usage

### 🛠️ Custom Configuration

Edit `config/settings.py` to customize:

```python
# Adjust analysis window
FEATURE_CONFIG = {
    "window_size": 45,  # More frames for greater stability
    "smooth_factor": 0.5,  # More smoothing
}

# Modify model parameters
MODEL_CONFIG = {
    "models": {
        "svm": {"C": 2.0, "kernel": "rbf"},  # More sensitive SVM
        "rf": {"n_estimators": 200},  # More robust Random Forest
    }
}
```

### 🎯 Training with Custom Data

1. **Add videos**: Place in `Videos/` folder with correct nomenclature
2. **Run training**: `python main.py train`
3. **Evaluate results**: Review metrics in `.json` files

### 🔌 Integration with Other Systems

```python
# Example of programmatic usage
from src.core.activity_classifier import ActivityClassifier
from src.core.feature_extractor import FeatureExtractor

classifier = ActivityClassifier("models/best_model.pkl")
extractor = FeatureExtractor()

# Process external data
features = extractor.extract_features()
activity, confidence, probs = classifier.predict(features)
print(f"Activity: {activity} (Confidence: {confidence:.2f})")
```

## 📞 Support and Contact

### 👥 Development Team
- **Davide Flamini** - [GitHub](https://github.com/davidone007)
- **Andrés Cabezas** - [GitHub](https://github.com/andrescabezas26)
- **Nicolas Cuellar** - [GitHub](https://github.com/Nicolas-CM)

### 🔗 Useful Links
- **Project Repository**: [PoseTrack_AI_ADN](https://github.com/Nicolas-CM/PoseTrack_AI_ADN.git)
- **MMFit Dataset**: [Official Site](https://mmfit.github.io)
- **MediaPipe**: [Documentation](https://developers.google.com/mediapipe)

### 🐛 Bug Reports
To report bugs or request features:
1. Create an issue in the GitHub repository
2. Include system information and error logs
3. Describe steps to reproduce the problem

---

