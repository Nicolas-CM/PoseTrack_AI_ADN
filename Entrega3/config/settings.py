"""
Global configuration for the PoseTrack AI system
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
VIDEOS_PATH = PROJECT_ROOT / "Videos"
MODELS_PATH = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"

# Create directories if they don't exist
MODELS_PATH.mkdir(exist_ok=True)
DATA_PATH.mkdir(exist_ok=True)
CONFIG_PATH.mkdir(exist_ok=True)

# MediaPipe configuration
MEDIAPIPE_CONFIG = {
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "enable_segmentation": False,
    "smooth_landmarks": True,
}

# Camera configuration
CAMERA_CONFIG = {"width": 640, "height": 480, "fps": 30}

# Basic movement activities
BASIC_ACTIVITIES = {
    "acercarse": "Approaching the camera",
    "alejarse": "Moving away from the camera",
    "girarD": "Turning right",
    "girarI": "Turning left",
    "sentarse": "Sitting down",
    "levantarse": "Standing up",
    "parado": "Standing still",
}

# Gym/exercise activities
GYM_ACTIVITIES = {
    "squat": "Squats",
    "russian_twist": "Russian twists",
    "push_up": "Push-ups",
}

# All activities (for compatibility)
ACTIVITIES = {**BASIC_ACTIVITIES, **GYM_ACTIVITIES}

# Feature configuration
FEATURE_CONFIG = {
    "window_size": 30,  # Sliding window of frames
    "normalize": True,
    "include_velocity": True,
    "include_angles": True,
    "smooth_factor": 0.3,
}

# Model configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "models": {
        "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        "rf": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "xgb": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
    },
}

# GUI configuration
GUI_CONFIG = {
    "window_title": "PoseTrack AI - Movement Analysis",
    "window_size": "1200x800",
    "video_size": (640, 480),
    "update_interval": 30,  # ms
}
