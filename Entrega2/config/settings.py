"""
Configuración global del sistema PoseTrack AI
"""

import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
VIDEOS_PATH = PROJECT_ROOT / "Videos"
MODELS_PATH = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"

# Crear directorios si no existen
MODELS_PATH.mkdir(exist_ok=True)
DATA_PATH.mkdir(exist_ok=True)
CONFIG_PATH.mkdir(exist_ok=True)

# Configuración de MediaPipe
MEDIAPIPE_CONFIG = {
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "enable_segmentation": False,
    "smooth_landmarks": True,
}

# Configuración de cámara
CAMERA_CONFIG = {"width": 640, "height": 480, "fps": 30}

# Actividades a detectar
ACTIVITIES = {
    "acercarse": "Acercándose a la cámara",
    "alejarse": "Alejándose de la cámara",
    "girarD": "Girando a la derecha",
    "girarI": "Girando a la izquierda",
    "sentarse": "Sentándose",
    "levantarse": "Levantándose",
}

# Configuración de características
FEATURE_CONFIG = {
    "window_size": 30,  # Ventana deslizante de frames
    "normalize": True,
    "include_velocity": True,
    "include_angles": True,
    "smooth_factor": 0.3,
}

# Configuración de modelos
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

# Configuración de GUI
GUI_CONFIG = {
    "window_title": "PoseTrack AI - Análisis de Movimiento",
    "window_size": "1200x800",
    "video_size": (640, 480),
    "update_interval": 30,  # ms
}
