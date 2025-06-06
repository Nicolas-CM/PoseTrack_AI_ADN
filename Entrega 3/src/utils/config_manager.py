"""
Gestión de configuraciones persistentes para PoseTrack AI
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

from config.settings import (
    CONFIG_PATH, 
    CAMERA_CONFIG, 
    MEDIAPIPE_CONFIG, 
    FEATURE_CONFIG,
    GUI_CONFIG
)


class ConfigManager:
    """Gestor de configuraciones persistentes"""
    
    def __init__(self):
        self.config_file = CONFIG_PATH / "user_settings.json"
        self._ensure_config_file()
    
    def _ensure_config_file(self):
        """Asegura que el archivo de configuración existe"""
        if not self.config_file.exists():
            self.save_default_config()
    
    def save_default_config(self):
        """Guarda la configuración por defecto"""
        default_config = {
            "camera": CAMERA_CONFIG.copy(),
            "mediapipe": MEDIAPIPE_CONFIG.copy(),
            "features": FEATURE_CONFIG.copy(),
            "gui": GUI_CONFIG.copy()
        }
        
        self._save_config(default_config)
    
    def _save_config(self, config: Dict[str, Any]):
        """Guarda la configuración en archivo JSON"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando configuración: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde archivo JSON"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error cargando configuración: {e}")
            return {}
    
    def save_camera_config(self, width: int, height: int, fps: int):
        """Guarda configuración de cámara"""
        config = self._load_config()
        
        if "camera" not in config:
            config["camera"] = {}
        
        config["camera"]["width"] = width
        config["camera"]["height"] = height
        config["camera"]["fps"] = fps
        
        self._save_config(config)
        
        # Actualizar configuración global
        CAMERA_CONFIG["width"] = width
        CAMERA_CONFIG["height"] = height
        CAMERA_CONFIG["fps"] = fps
    
    def save_mediapipe_config(self, min_detection_confidence: float, 
                             model_complexity: int, min_tracking_confidence: float = None):
        """Guarda configuración de MediaPipe"""
        config = self._load_config()
        
        if "mediapipe" not in config:
            config["mediapipe"] = {}
        
        config["mediapipe"]["min_detection_confidence"] = min_detection_confidence
        config["mediapipe"]["model_complexity"] = model_complexity
        
        if min_tracking_confidence is not None:
            config["mediapipe"]["min_tracking_confidence"] = min_tracking_confidence
        
        self._save_config(config)
        
        # Actualizar configuración global
        MEDIAPIPE_CONFIG["min_detection_confidence"] = min_detection_confidence
        MEDIAPIPE_CONFIG["model_complexity"] = model_complexity
        if min_tracking_confidence is not None:
            MEDIAPIPE_CONFIG["min_tracking_confidence"] = min_tracking_confidence
    
    def save_feature_config(self, window_size: int, buffer_size: int = None):
        """Guarda configuración de características"""
        config = self._load_config()
        
        if "features" not in config:
            config["features"] = {}
        
        config["features"]["window_size"] = window_size
        
        if buffer_size is not None:
            config["features"]["buffer_size"] = buffer_size
        
        self._save_config(config)
        
        # Actualizar configuración global
        FEATURE_CONFIG["window_size"] = window_size
        if buffer_size is not None and "buffer_size" in FEATURE_CONFIG:
            FEATURE_CONFIG["buffer_size"] = buffer_size
    
    def load_all_configs(self):
        """Carga todas las configuraciones y actualiza las globales"""
        config = self._load_config()
        
        if "camera" in config:
            CAMERA_CONFIG.update(config["camera"])
        
        if "mediapipe" in config:
            MEDIAPIPE_CONFIG.update(config["mediapipe"])
        
        if "features" in config:
            FEATURE_CONFIG.update(config["features"])
        
        if "gui" in config:
            GUI_CONFIG.update(config["gui"])
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Obtiene configuración de cámara"""
        config = self._load_config()
        return config.get("camera", CAMERA_CONFIG)
    
    def get_mediapipe_config(self) -> Dict[str, Any]:
        """Obtiene configuración de MediaPipe"""
        config = self._load_config()
        return config.get("mediapipe", MEDIAPIPE_CONFIG)
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Obtiene configuración de características"""
        config = self._load_config()
        return config.get("features", FEATURE_CONFIG)
    
    def reset_to_defaults(self):
        """Resetea todas las configuraciones a los valores por defecto"""
        self.save_default_config()
        self.load_all_configs()


# Instancia global del gestor de configuraciones
config_manager = ConfigManager()
