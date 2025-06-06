"""
Módulo de clasificación de actividades en tiempo real
"""

import numpy as np
import joblib
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json

from config.settings import MODEL_CONFIG, ACTIVITIES, MODELS_PATH


class ActivityClassifier:
    """Clasificador de actividades humanas en tiempo real"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_name = None
        self.feature_scaler = None
        self.label_encoder = None
        self.model_info = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Carga un modelo entrenado desde archivo
        
        Args:
            model_path: Ruta al archivo del modelo
            
        Returns:
            True si la carga fue exitosa, False en caso contrario
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                print(f"Error: El archivo {model_path} no existe")
                return False
            
            # Cargar modelo principal
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.feature_scaler = model_data.get('scaler')
                self.label_encoder = model_data.get('label_encoder')
                self.model_info = model_data.get('info', {})
                self.model_name = self.model_info.get('model_type', 'unknown')
            else:
                # Retrocompatibilidad: modelo sin metadatos
                self.model = model_data
                self.model_name = 'legacy'
            
            print(f"Modelo {self.model_name} cargado exitosamente")
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predice la actividad para un conjunto de características
        
        Args:
            features: Array de características extraídas
            
        Returns:
            Tupla con (actividad_predicha, confianza, probabilidades_por_clase)
        """
        if self.model is None:
            return "sin_modelo", 0.0, {}
        
        try:
            # Asegurar que features sea 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Normalizar características si hay scaler
            if self.feature_scaler is not None:
                features = self.feature_scaler.transform(features)
            
            # Realizar predicción
            prediction = self.model.predict(features)[0]
            
            # Obtener probabilidades si es posible
            probabilities = {}
            confidence = 0.0
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                confidence = np.max(proba)
                
                # Mapear probabilidades a nombres de clase
                if self.label_encoder is not None:
                    classes = self.label_encoder.classes_
                else:
                    classes = list(ACTIVITIES.keys())
                
                for i, class_name in enumerate(classes):
                    if i < len(proba):
                        probabilities[class_name] = float(proba[i])
            else:
                # Para modelos sin predict_proba, usar confianza fija
                confidence = 0.8
                probabilities[prediction] = confidence
            
            # Decodificar etiqueta si hay label encoder
            if self.label_encoder is not None:
                try:
                    predicted_activity = self.label_encoder.inverse_transform([prediction])[0]
                except:
                    predicted_activity = str(prediction)
            else:
                predicted_activity = str(prediction)
            
            return predicted_activity, confidence, probabilities
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return "error", 0.0, {}
    
    def predict_batch(self, features_batch: np.ndarray) -> List[Tuple[str, float]]:
        """
        Predice actividades para un lote de características
        
        Args:
            features_batch: Array 2D con características de múltiples muestras
            
        Returns:
            Lista de tuplas (actividad, confianza)
        """
        if self.model is None:
            return [("sin_modelo", 0.0)] * len(features_batch)
        
        results = []
        
        try:
            # Normalizar si hay scaler
            if self.feature_scaler is not None:
                features_batch = self.feature_scaler.transform(features_batch)
            
            # Realizar predicciones
            predictions = self.model.predict(features_batch)
            
            # Obtener probabilidades si es posible
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_batch)
                confidences = np.max(probabilities, axis=1)
            else:
                confidences = [0.8] * len(predictions)
            
            # Procesar resultados
            for pred, conf in zip(predictions, confidences):
                if self.label_encoder is not None:
                    try:
                        activity = self.label_encoder.inverse_transform([pred])[0]
                    except:
                        activity = str(pred)
                else:
                    activity = str(pred)
                
                results.append((activity, float(conf)))
            
        except Exception as e:
            print(f"Error en predicción por lotes: {e}")
            results = [("error", 0.0)] * len(features_batch)
        
        return results
    
    def get_activity_description(self, activity_code: str) -> str:
        """
        Obtiene la descripción legible de una actividad
        
        Args:
            activity_code: Código de la actividad
            
        Returns:
            Descripción de la actividad
        """
        return ACTIVITIES.get(activity_code, f"Actividad: {activity_code}")
    
    def get_model_info(self) -> Dict:
        """
        Retorna información sobre el modelo cargado
        
        Returns:
            Diccionario con información del modelo
        """
        info = {
            'model_loaded': self.model is not None,
            'model_name': self.model_name,
            'has_scaler': self.feature_scaler is not None,
            'has_label_encoder': self.label_encoder is not None,
            'supported_activities': list(ACTIVITIES.keys())
        }
        
        info.update(self.model_info)
        return info
    
    def is_ready(self) -> bool:
        """
        Verifica si el clasificador está listo para hacer predicciones
        
        Returns:
            True si el modelo está cargado y listo
        """
        return self.model is not None
    
    def get_available_models(self) -> List[Dict]:
        """
        Obtiene lista de modelos disponibles en el directorio de modelos
        
        Returns:
            Lista de diccionarios con información de modelos disponibles
        """
        models = []
        models_dir = Path(MODELS_PATH)
        
        if not models_dir.exists():
            return models
        
        # Buscar archivos de modelo
        for model_file in models_dir.glob("*.pkl"):
            try:
                # Intentar cargar información básica
                model_data = joblib.load(model_file)
                
                if isinstance(model_data, dict) and 'info' in model_data:
                    info = model_data['info']
                    models.append({
                        'path': str(model_file),
                        'name': model_file.stem,
                        'type': info.get('model_type', 'unknown'),
                        'accuracy': info.get('accuracy', 'N/A'),
                        'created': info.get('created_at', 'N/A'),
                        'features': info.get('n_features', 'N/A')
                    })
                else:
                    # Modelo legacy sin metadatos
                    models.append({
                        'path': str(model_file),
                        'name': model_file.stem,
                        'type': 'legacy',
                        'accuracy': 'N/A',
                        'created': 'N/A',
                        'features': 'N/A'
                    })
                    
            except Exception as e:
                print(f"Error leyendo modelo {model_file}: {e}")
        
        return models
    
    def clear_model(self):
        """Limpia el modelo cargado"""
        self.model = None
        self.model_name = None
        self.feature_scaler = None
        self.label_encoder = None
        self.model_info = {}


class ActivityBuffer:
    """Buffer para suavizar predicciones de actividades"""
    
    def __init__(self, buffer_size: int = 10, confidence_threshold: float = 0.6):
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.predictions = []
        self.confidences = []
    
    def add_prediction(self, activity: str, confidence: float):
        """
        Añade una nueva predicción al buffer
        
        Args:
            activity: Actividad predicha
            confidence: Confianza de la predicción
        """
        self.predictions.append(activity)
        self.confidences.append(confidence)
        
        # Mantener tamaño del buffer
        if len(self.predictions) > self.buffer_size:
            self.predictions.pop(0)
            self.confidences.pop(0)
    
    def get_smoothed_prediction(self) -> Tuple[str, float]:
        """
        Obtiene la predicción suavizada
        
        Returns:
            Tupla con (actividad_suavizada, confianza_promedio)
        """
        if not self.predictions:
            return "desconocido", 0.0
        
        # Contar frecuencias
        activity_counts = {}
        activity_confidences = {}
        
        for activity, confidence in zip(self.predictions, self.confidences):
            if confidence >= self.confidence_threshold:
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
                if activity not in activity_confidences:
                    activity_confidences[activity] = []
                activity_confidences[activity].append(confidence)
        
        if not activity_counts:
            return "incierto", np.mean(self.confidences)
        
        # Actividad más frecuente
        most_common = max(activity_counts.items(), key=lambda x: x[1])
        best_activity = most_common[0]
        
        # Confianza promedio para esa actividad
        avg_confidence = np.mean(activity_confidences[best_activity])
        
        return best_activity, avg_confidence
    
    def clear(self):
        """Limpia el buffer"""
        self.predictions.clear()
        self.confidences.clear()
