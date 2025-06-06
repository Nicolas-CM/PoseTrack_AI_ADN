"""
Módulo de extracción y normalización de características
"""

import numpy as np
from typing import List, Dict, Optional
from collections import deque
import pandas as pd
from scipy.signal import savgol_filter

from config.settings import FEATURE_CONFIG


class FeatureExtractor:
    """Extrae características temporales de secuencias de landmarks"""
    
    def __init__(self, window_size: int = None):
        self.window_size = window_size or FEATURE_CONFIG['window_size']
        self.landmarks_buffer = deque(maxlen=self.window_size)
        self.angles_buffer = deque(maxlen=self.window_size)
        self.timestamps_buffer = deque(maxlen=self.window_size)
        
        # Estadísticas para normalización
        self.stats = None
        
    def add_frame_data(self, landmarks: List[float], angles: Dict[str, float], 
                      timestamp: float = None):
        """
        Añade datos de un frame al buffer
        
        Args:
            landmarks: Lista de landmarks normalizados
            angles: Diccionario de ángulos calculados
            timestamp: Timestamp del frame (opcional)
        """
        if landmarks:
            self.landmarks_buffer.append(landmarks)
            self.angles_buffer.append(angles)
            self.timestamps_buffer.append(timestamp or len(self.landmarks_buffer))
    
    def extract_features(self) -> Optional[np.ndarray]:
        """
        Extrae características de la ventana actual
        
        Returns:
            Array de características o None si no hay suficientes datos
        """
        if len(self.landmarks_buffer) < self.window_size:
            return None
        
        features = []
        
        # 1. Características estadísticas de landmarks
        landmarks_matrix = np.array(list(self.landmarks_buffer))
        features.extend(self._extract_statistical_features(landmarks_matrix))
        
        # 2. Características de velocidad
        if FEATURE_CONFIG['include_velocity']:
            features.extend(self._extract_velocity_features(landmarks_matrix))
        
        # 3. Características de ángulos
        if FEATURE_CONFIG['include_angles']:
            angles_matrix = self._angles_to_matrix()
            features.extend(self._extract_angle_features(angles_matrix))
        
        # 4. Características de trayectoria
        features.extend(self._extract_trajectory_features(landmarks_matrix))
        
        # 5. Características de frecuencia
        features.extend(self._extract_frequency_features(landmarks_matrix))
        
        return np.array(features)
    
    def _extract_statistical_features(self, landmarks_matrix: np.ndarray) -> List[float]:
        """Extrae características estadísticas básicas"""
        features = []
        
        # Para cada landmark (coordenadas x, y, z)
        for i in range(0, landmarks_matrix.shape[1], 4):  # Saltar visibility
            if i + 2 < landmarks_matrix.shape[1]:
                x_coords = landmarks_matrix[:, i]
                y_coords = landmarks_matrix[:, i + 1]
                z_coords = landmarks_matrix[:, i + 2]
                
                # Estadísticas para cada coordenada
                for coords in [x_coords, y_coords, z_coords]:
                    features.extend([
                        np.mean(coords),
                        np.std(coords),
                        np.min(coords),
                        np.max(coords),
                        np.median(coords)
                    ])
        
        return features
    
    def _extract_velocity_features(self, landmarks_matrix: np.ndarray) -> List[float]:
        """Extrae características de velocidad"""
        features = []
        
        if landmarks_matrix.shape[0] < 2:
            return [0.0] * 50  # Placeholder
        
        # Calcular velocidades
        velocities = np.diff(landmarks_matrix, axis=0)
        
        # Estadísticas de velocidad para puntos clave
        key_points = [0, 4, 8, 12, 16, 20, 24, 28]  # Indices de puntos importantes
        
        for point_idx in key_points:
            base_idx = point_idx * 4
            if base_idx + 2 < velocities.shape[1]:
                vx = velocities[:, base_idx]
                vy = velocities[:, base_idx + 1]
                
                # Magnitud de velocidad
                speed = np.sqrt(vx**2 + vy**2)
                
                features.extend([
                    np.mean(speed),
                    np.std(speed),
                    np.max(speed)
                ])
        
        return features
    
    def _extract_angle_features(self, angles_matrix: np.ndarray) -> List[float]:
        """Extrae características de ángulos articulares"""
        features = []
        
        if angles_matrix.size == 0:
            return [0.0] * 30  # Placeholder
        
        # Para cada tipo de ángulo
        for col in range(angles_matrix.shape[1]):
            angle_series = angles_matrix[:, col]
            
            # Filtrar valores NaN
            valid_angles = angle_series[~np.isnan(angle_series)]
            
            if len(valid_angles) > 0:
                features.extend([
                    np.mean(valid_angles),
                    np.std(valid_angles),
                    np.min(valid_angles),
                    np.max(valid_angles),
                    np.ptp(valid_angles)  # Peak to peak (rango)
                ])
            else:
                features.extend([0.0] * 5)
        
        return features
    
    def _extract_trajectory_features(self, landmarks_matrix: np.ndarray) -> List[float]:
        """Extrae características de trayectoria"""
        features = []
        
        # Puntos clave para trayectoria (nariz, manos, pies)
        key_points = {
            'nose': 0, 'left_wrist': 60, 'right_wrist': 64,
            'left_ankle': 108, 'right_ankle': 112
        }
        
        for point_name, base_idx in key_points.items():
            if base_idx + 1 < landmarks_matrix.shape[1]:
                x_coords = landmarks_matrix[:, base_idx]
                y_coords = landmarks_matrix[:, base_idx + 1]
                
                # Distancia total recorrida
                distances = np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2)
                total_distance = np.sum(distances)
                
                # Desplazamiento neto
                net_displacement = np.sqrt(
                    (x_coords[-1] - x_coords[0])**2 + 
                    (y_coords[-1] - y_coords[0])**2
                )
                
                # Relación desplazamiento/distancia (rectitud de trayectoria)
                straightness = net_displacement / (total_distance + 1e-8)
                
                features.extend([total_distance, net_displacement, straightness])
        
        return features
    
    def _extract_frequency_features(self, landmarks_matrix: np.ndarray) -> List[float]:
        """Extrae características de frecuencia"""
        features = []
        
        # Análisis de frecuencia para puntos clave
        key_indices = [0, 4, 60, 64, 108, 112]  # nose, shoulders, wrists, ankles
        
        for idx in key_indices:
            if idx + 1 < landmarks_matrix.shape[1]:
                # Coordenada Y (movimiento vertical)
                y_coords = landmarks_matrix[:, idx + 1]
                
                # Suavizar señal
                if len(y_coords) > 5:
                    smoothed = savgol_filter(y_coords, 5, 2)
                else:
                    smoothed = y_coords
                
                # FFT para encontrar frecuencias dominantes
                fft = np.fft.fft(smoothed)
                freqs = np.fft.fftfreq(len(smoothed))
                
                # Energía en diferentes bandas de frecuencia
                low_freq_energy = np.sum(np.abs(fft[np.abs(freqs) < 0.1]))
                mid_freq_energy = np.sum(np.abs(fft[(np.abs(freqs) >= 0.1) & (np.abs(freqs) < 0.3)]))
                high_freq_energy = np.sum(np.abs(fft[np.abs(freqs) >= 0.3]))
                
                features.extend([low_freq_energy, mid_freq_energy, high_freq_energy])
        
        return features
    
    def _angles_to_matrix(self) -> np.ndarray:
        """Convierte el buffer de ángulos a matriz numpy"""
        if not self.angles_buffer:
            return np.array([])
        
        # Obtener todas las claves de ángulos
        angle_keys = list(self.angles_buffer[0].keys()) if self.angles_buffer[0] else []
        
        if not angle_keys:
            return np.array([])
        
        # Crear matriz
        matrix = []
        for angles_dict in self.angles_buffer:
            row = [angles_dict.get(key, np.nan) for key in angle_keys]
            matrix.append(row)
        
        return np.array(matrix)
    
    def normalize_features(self, features: np.ndarray, 
                          fit: bool = False) -> np.ndarray:
        """
        Normaliza las características
        
        Args:
            features: Array de características
            fit: Si True, calcula estadísticas de normalización
            
        Returns:
            Características normalizadas
        """
        if not FEATURE_CONFIG['normalize']:
            return features
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if fit or self.stats is None:
            self.stats = {
                'mean': np.mean(features, axis=0),
                'std': np.std(features, axis=0) + 1e-8  # Evitar división por cero
            }
        
        normalized = (features - self.stats['mean']) / self.stats['std']
        return normalized
    
    def get_feature_names(self) -> List[str]:
        """Retorna los nombres de las características extraídas"""
        names = []
        
        # Nombres de landmarks
        landmark_names = ['nose', 'left_eye', 'right_eye', 'left_shoulder', 
                         'right_shoulder', 'left_elbow', 'right_elbow',
                         'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                         'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        # Características estadísticas
        for landmark in landmark_names:
            for coord in ['x', 'y', 'z']:
                for stat in ['mean', 'std', 'min', 'max', 'median']:
                    names.append(f"{landmark}_{coord}_{stat}")
        
        # Características de velocidad
        if FEATURE_CONFIG['include_velocity']:
            for landmark in ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow',
                           'right_elbow', 'left_wrist', 'right_wrist', 'left_hip']:
                for stat in ['speed_mean', 'speed_std', 'speed_max']:
                    names.append(f"{landmark}_{stat}")
        
        # Características de ángulos
        if FEATURE_CONFIG['include_angles']:
            angle_names = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee',
                          'trunk_inclination', 'left_hip', 'right_hip']
            for angle in angle_names:
                for stat in ['mean', 'std', 'min', 'max', 'range']:
                    names.append(f"angle_{angle}_{stat}")
        
        return names
    
    def reset_buffer(self):
        """Limpia todos los buffers"""
        self.landmarks_buffer.clear()
        self.angles_buffer.clear()
        self.timestamps_buffer.clear()
    
    def is_ready(self) -> bool:
        """Verifica si hay suficientes datos para extraer características"""
        return len(self.landmarks_buffer) >= self.window_size
