"""
Módulo de seguimiento de pose usando MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List
import math

from config.settings import MEDIAPIPE_CONFIG


class PoseTracker:
    """Clase para el seguimiento de pose en tiempo real usando MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=MEDIAPIPE_CONFIG['model_complexity'],
            min_detection_confidence=MEDIAPIPE_CONFIG['min_detection_confidence'],
            min_tracking_confidence=MEDIAPIPE_CONFIG['min_tracking_confidence'],
            enable_segmentation=MEDIAPIPE_CONFIG['enable_segmentation'],
            smooth_landmarks=MEDIAPIPE_CONFIG['smooth_landmarks']
        )
        
        # Landmarks importantes para análisis
        self.key_landmarks = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[List]]:
        """
        Procesa un frame y extrae los landmarks de pose
        
        Args:
            frame: Frame de video en formato BGR
            
        Returns:
            Tuple con frame procesado y landmarks extraídos
        """
        # Convertir BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Procesar con MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Convertir de vuelta a BGR
        rgb_frame.flags.writeable = True
        annotated_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        landmarks = None
        if results.pose_landmarks:
            # Dibujar landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Extraer coordenadas normalizadas
            landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
        
        return annotated_frame, landmarks
    
    def _extract_landmarks(self, pose_landmarks, frame_shape) -> List[float]:
        """
        Extrae las coordenadas de los landmarks clave
        
        Args:
            pose_landmarks: Landmarks de MediaPipe
            frame_shape: Dimensiones del frame (height, width, channels)
            
        Returns:
            Lista de coordenadas normalizadas [x1, y1, z1, x2, y2, z2, ...]
        """
        landmarks = []
        height, width = frame_shape[:2]
        
        for landmark_name in self.key_landmarks:
            idx = self.key_landmarks[landmark_name]
            landmark = pose_landmarks.landmark[idx]
            
            # Normalizar coordenadas
            x = landmark.x
            y = landmark.y
            z = landmark.z
            visibility = landmark.visibility
            
            landmarks.extend([x, y, z, visibility])
        
        return landmarks
    
    def calculate_angles(self, landmarks: List[float]) -> dict:
        """
        Calcula ángulos importantes entre articulaciones
        
        Args:
            landmarks: Lista de landmarks normalizados
            
        Returns:
            Diccionario con ángulos calculados
        """
        if not landmarks or len(landmarks) < len(self.key_landmarks) * 4:
            return {}
        
        # Convertir landmarks a diccionario de coordenadas
        points = {}
        for i, name in enumerate(self.key_landmarks.keys()):
            base_idx = i * 4
            points[name] = {
                'x': landmarks[base_idx],
                'y': landmarks[base_idx + 1],
                'z': landmarks[base_idx + 2],
                'visibility': landmarks[base_idx + 3]
            }
        
        angles = {}
        
        try:
            # Ángulo del codo izquierdo
            angles['left_elbow'] = self._calculate_angle(
                points['left_shoulder'], points['left_elbow'], points['left_wrist']
            )
            
            # Ángulo del codo derecho
            angles['right_elbow'] = self._calculate_angle(
                points['right_shoulder'], points['right_elbow'], points['right_wrist']
            )
            
            # Ángulo de la rodilla izquierda
            angles['left_knee'] = self._calculate_angle(
                points['left_hip'], points['left_knee'], points['left_ankle']
            )
            
            # Ángulo de la rodilla derecha
            angles['right_knee'] = self._calculate_angle(
                points['right_hip'], points['right_knee'], points['right_ankle']
            )
            
            # Inclinación del tronco (ángulo entre hombros y cadera)
            angles['trunk_inclination'] = self._calculate_trunk_inclination(
                points['left_shoulder'], points['right_shoulder'],
                points['left_hip'], points['right_hip']
            )
            
            # Ángulo de la cadera izquierda
            angles['left_hip'] = self._calculate_angle(
                points['left_shoulder'], points['left_hip'], points['left_knee']
            )
            
            # Ángulo de la cadera derecha
            angles['right_hip'] = self._calculate_angle(
                points['right_shoulder'], points['right_hip'], points['right_knee']
            )
            
        except Exception as e:
            print(f"Error calculando ángulos: {e}")
        
        return angles
    
    def _calculate_angle(self, point1: dict, point2: dict, point3: dict) -> float:
        """
        Calcula el ángulo entre tres puntos
        
        Args:
            point1, point2, point3: Puntos con coordenadas x, y, z
            
        Returns:
            Ángulo en grados
        """
        # Vectores
        v1 = np.array([point1['x'] - point2['x'], point1['y'] - point2['y']])
        v2 = np.array([point3['x'] - point2['x'], point3['y'] - point2['y']])
        
        # Calcular ángulo
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Evitar errores numéricos
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _calculate_trunk_inclination(self, left_shoulder: dict, right_shoulder: dict,
                                   left_hip: dict, right_hip: dict) -> float:
        """
        Calcula la inclinación del tronco
        
        Returns:
            Ángulo de inclinación en grados
        """
        # Punto medio de hombros y caderas
        shoulder_mid = {
            'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
            'y': (left_shoulder['y'] + right_shoulder['y']) / 2
        }
        
        hip_mid = {
            'x': (left_hip['x'] + right_hip['x']) / 2,
            'y': (left_hip['y'] + right_hip['y']) / 2
        }
        
        # Vector del tronco
        trunk_vector = np.array([
            shoulder_mid['x'] - hip_mid['x'],
            shoulder_mid['y'] - hip_mid['y']
        ])
        
        # Vector vertical de referencia
        vertical_vector = np.array([0, -1])  # Hacia arriba en coordenadas de imagen
        
        # Calcular ángulo
        cos_angle = np.dot(trunk_vector, vertical_vector) / (
            np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector)
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def get_pose_confidence(self, landmarks: List[float]) -> float:
        """
        Calcula la confianza promedio de la detección de pose
        
        Args:
            landmarks: Lista de landmarks con visibility
            
        Returns:
            Confianza promedio (0-1)
        """
        if not landmarks:
            return 0.0
        
        # Extraer valores de visibility (cada 4to elemento)
        visibilities = [landmarks[i] for i in range(3, len(landmarks), 4)]
        
        return np.mean(visibilities) if visibilities else 0.0
    
    def close(self):
        """Libera recursos de MediaPipe"""
        try:
            if hasattr(self, 'pose') and self.pose is not None:
                self.pose.close()
                self.pose = None
        except Exception as e:
            # Silenciar errores de cleanup
            pass
