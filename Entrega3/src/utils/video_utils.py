"""
Video and data processing utilities

This module provides helper functions for working with video files,
including validation, information extraction, frame processing,
and video encoding/decoding operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json

def validate_video_file(video_path: str) -> bool:
    """
    Validate if a video file is valid and can be opened
    
    This function attempts to open a video file and read the first frame
    to verify that the file is a valid and readable video.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if the video is valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Intentar leer el primer frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    except:
        return False

def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file
    
    This function extracts metadata from a video file including resolution,
    frame rate, duration, and frame count. It's useful for analyzing and
    preprocessing video content.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video information
    """
    info = {
        'path': video_path,
        'valid': False,
        'fps': 0,
        'frame_count': 0,
        'duration': 0,
        'width': 0,
        'height': 0,
        'size_mb': 0
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            info['valid'] = True
            info['fps'] = cap.get(cv2.CAP_PROP_FPS)
            info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if info['fps'] > 0:
                info['duration'] = info['frame_count'] / info['fps']
            
            cap.release()
        
        # Tamaño del archivo
        file_path = Path(video_path)
        if file_path.exists():
            info['size_mb'] = file_path.stat().st_size / (1024 * 1024)
    
    except Exception as e:
        print(f"Error obteniendo información del video {video_path}: {e}")
    
    return info

def resize_frame_keep_aspect(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Redimensiona un frame manteniendo la relación de aspecto
    
    Args:
        frame: Frame de video
        target_size: Tamaño objetivo (width, height)
        
    Returns:
        Frame redimensionado
    """
    target_width, target_height = target_size
    height, width = frame.shape[:2]
    
    # Calcular escalas
    scale_w = target_width / width
    scale_h = target_height / height
    scale = min(scale_w, scale_h)
    
    # Nuevo tamaño
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Redimensionar
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Crear frame con padding si es necesario
    if new_width != target_width or new_height != target_height:
        # Calcular padding
        pad_w = (target_width - new_width) // 2
        pad_h = (target_height - new_height) // 2
        
        # Crear frame con padding negro
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded[pad_h:pad_h+new_height, pad_w:pad_w+new_width] = resized
        
        return padded
    
    return resized

def smooth_landmarks(landmarks_sequence: List[List[float]], 
                    window_size: int = 5) -> List[List[float]]:
    """
    Suaviza una secuencia de landmarks usando media móvil
    
    Args:
        landmarks_sequence: Secuencia de landmarks
        window_size: Tamaño de ventana para suavizado
        
    Returns:
        Secuencia de landmarks suavizada
    """
    if len(landmarks_sequence) < window_size:
        return landmarks_sequence
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(landmarks_sequence)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(landmarks_sequence), i + half_window + 1)
        
        # Calcular media de la ventana
        window_landmarks = landmarks_sequence[start_idx:end_idx]
        
        if window_landmarks:
            # Transponer para calcular media por coordenada
            landmarks_array = np.array(window_landmarks)
            mean_landmarks = np.mean(landmarks_array, axis=0)
            smoothed.append(mean_landmarks.tolist())
        else:
            smoothed.append(landmarks_sequence[i])
    
    return smoothed

def calculate_motion_intensity(landmarks_sequence: List[List[float]]) -> float:
    """
    Calcula la intensidad de movimiento en una secuencia
    
    Args:
        landmarks_sequence: Secuencia de landmarks
        
    Returns:
        Intensidad de movimiento (0-1)
    """
    if len(landmarks_sequence) < 2:
        return 0.0
    
    total_movement = 0.0
    num_comparisons = 0
    
    for i in range(1, len(landmarks_sequence)):
        prev_landmarks = np.array(landmarks_sequence[i-1])
        curr_landmarks = np.array(landmarks_sequence[i])
        
        # Calcular diferencias (solo coordenadas x, y, z - saltar visibility)
        differences = []
        for j in range(0, len(prev_landmarks), 4):
            if j + 2 < len(prev_landmarks):
                prev_point = prev_landmarks[j:j+3]
                curr_point = curr_landmarks[j:j+3]
                
                # Distancia euclidiana
                distance = np.linalg.norm(curr_point - prev_point)
                differences.append(distance)
        
        if differences:
            frame_movement = np.mean(differences)
            total_movement += frame_movement
            num_comparisons += 1
    
    if num_comparisons > 0:
        avg_movement = total_movement / num_comparisons
        # Normalizar (ajustar según datos experimentales)
        normalized_movement = min(avg_movement * 10, 1.0)
        return normalized_movement
    
    return 0.0

def extract_keypoints_subset(landmarks: List[float], 
                           keypoint_indices: List[int]) -> List[float]:
    """
    Extrae un subconjunto de keypoints de la lista completa
    
    Args:
        landmarks: Lista completa de landmarks
        keypoint_indices: Índices de los keypoints a extraer
        
    Returns:
        Lista con solo los keypoints seleccionados
    """
    subset = []
    
    for idx in keypoint_indices:
        base_idx = idx * 4  # 4 valores por landmark (x, y, z, visibility)
        if base_idx + 3 < len(landmarks):
            subset.extend(landmarks[base_idx:base_idx+4])
    
    return subset

def save_processed_data(data: dict, filepath: str):
    """
    Guarda datos procesados en formato JSON
    
    Args:
        data: Datos a guardar
        filepath: Ruta del archivo
    """
    try:
        # Convertir arrays numpy a listas para JSON
        json_data = convert_numpy_to_list(data)
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Datos guardados en: {filepath}")
    
    except Exception as e:
        print(f"Error guardando datos: {e}")

def load_processed_data(filepath: str) -> Optional[dict]:
    """
    Carga datos procesados desde JSON
    
    Args:
        filepath: Ruta del archivo
        
    Returns:
        Datos cargados o None si hay error
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"Datos cargados desde: {filepath}")
        return data
    
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None

def convert_numpy_to_list(obj):
    """
    Convierte recursivamente arrays numpy a listas para serialización JSON
    
    Args:
        obj: Objeto a convertir
        
    Returns:
        Objeto con arrays convertidos a listas
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    else:
        return obj

def create_video_summary(videos_directory: str) -> dict:
    """
    Crea un resumen de todos los videos en un directorio
    
    Args:
        videos_directory: Directorio con videos
        
    Returns:
        Diccionario con resumen de videos
    """
    videos_dir = Path(videos_directory)
    summary = {
        'directory': str(videos_dir),
        'total_videos': 0,
        'valid_videos': 0,
        'total_duration': 0,
        'total_size_mb': 0,
        'videos': []
    }
    
    if not videos_dir.exists():
        return summary
    
    # Extensiones de video soportadas
    video_extensions = ['.mp4', '.avi', '.mov', '.MOV', '.mkv', '.flv']
    
    for ext in video_extensions:
        for video_file in videos_dir.glob(f"*{ext}"):
            summary['total_videos'] += 1
            
            info = get_video_info(str(video_file))
            summary['videos'].append(info)
            
            if info['valid']:
                summary['valid_videos'] += 1
                summary['total_duration'] += info['duration']
                summary['total_size_mb'] += info['size_mb']
    
    return summary
