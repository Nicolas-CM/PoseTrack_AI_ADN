"""
Sistema de entrenamiento de modelos para clasificación de actividades
"""

import cv2
i        patterns = {
            'acercarse': r'acerca.*',
            'alejarse': r'aleja.*',
            'girarD': r'girar.*d.*',
            'girarI': r'girar.*i.*',
            'sentarse': r'sent.*',
            'levantarse': r'levant.*',
            'parado': r'parado.*',
            'squat': r'squat.*',
            'barbell_biceps_curl': r'barbell.*biceps.*curl.*',
            'push_up': r'push.*up.*'
        }py as np
import pandas as pd
from pathlib import Path
import joblib
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

from config.settings import VIDEOS_PATH, MODELS_PATH, DATA_PATH, MODEL_CONFIG, ACTIVITIES
from src.core.pose_tracker import PoseTracker
from src.core.feature_extractor import FeatureExtractor


class VideoDataProcessor:
    """Procesa videos para extraer datos de entrenamiento"""
    
    def __init__(self):
        self.pose_tracker = PoseTracker()
        self.feature_extractor = FeatureExtractor()
        
    def process_video(self, video_path: str, activity_label: str) -> List[np.ndarray]:
        """
        Procesa un video y extrae características
        
        Args:
            video_path: Ruta al video
            activity_label: Etiqueta de la actividad
            
        Returns:
            Lista de arrays de características
        """
        features_list = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return features_list
        
        frame_count = 0
        self.feature_extractor.reset_buffer()
        
        print(f"Procesando: {Path(video_path).name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame con MediaPipe
            annotated_frame, landmarks = self.pose_tracker.process_frame(frame)
            
            if landmarks:
                # Calcular ángulos
                angles = self.pose_tracker.calculate_angles(landmarks)
                
                # Añadir al buffer del extractor de características
                self.feature_extractor.add_frame_data(landmarks, angles, frame_count)
                
                # Extraer características si el buffer está lleno
                if self.feature_extractor.is_ready():
                    features = self.feature_extractor.extract_features()
                    if features is not None:
                        features_list.append(features)
            
            frame_count += 1
        
        cap.release()
        print(f"Extraídas {len(features_list)} secuencias de características")
        
        return features_list
    
    def extract_activity_from_filename(self, filename: str) -> Optional[str]:
        """
        Extrae la etiqueta de actividad del nombre del archivo
        
        Args:
            filename: Nombre del archivo de video
            
        Returns:
            Etiqueta de actividad o None si no se puede determinar
        """
        filename = filename.lower()
          # Patrones para diferentes actividades
        patterns = {
            'acercarse': r'acer.*',
            'alejarse': r'alej.*',
            'girarD': r'girar.*d.*',
            'girarI': r'girar.*i.*',
            'sentarse': r'sent.*',
            'levantarse': r'levant.*',
            'parado': r'parado.*'
        }
        
        for activity, pattern in patterns.items():
            if re.search(pattern, filename):
                return activity
        
        return None


class ModelTrainer:
    """Entrenador de modelos de clasificación de actividades"""
    
    def __init__(self):
        self.data_processor = VideoDataProcessor()
        self.models = {
            'svm': SVC(**MODEL_CONFIG['models']['svm'], probability=True),
            'rf': RandomForestClassifier(**MODEL_CONFIG['models']['rf']),
            'xgb': xgb.XGBClassifier(**MODEL_CONFIG['models']['xgb'])
        }
        
    def prepare_training_data(self, videos_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos de entrenamiento desde videos
        
        Args:
            videos_dir: Directorio con videos de entrenamiento
            
        Returns:
            Tupla con (características, etiquetas)
        """
        if videos_dir is None:
            videos_dir = VIDEOS_PATH
        
        videos_dir = Path(videos_dir)
        
        if not videos_dir.exists():
            raise ValueError(f"El directorio {videos_dir} no existe")
        
        all_features = []
        all_labels = []
          # Buscar videos
        video_extensions = ['.mp4', '.avi', '.mov', '.MOV']
        video_files = []
        
        # Buscar en el directorio principal
        for ext in video_extensions:
            video_files.extend(videos_dir.glob(f"*{ext}"))
        
        # Buscar también en subdirectorios (como Gym)
        for ext in video_extensions:
            video_files.extend(videos_dir.glob(f"**/*{ext}"))
        
        # Eliminar duplicados
        video_files = list(set(video_files))
        
        if not video_files:
            raise ValueError(f"No se encontraron videos en {videos_dir}")
        
        print(f"Encontrados {len(video_files)} videos para procesar")
        
        for video_file in tqdm(video_files, desc="Procesando videos"):
            # Extraer etiqueta del nombre del archivo
            activity = self.data_processor.extract_activity_from_filename(video_file.name)
            
            if activity is None:
                print(f"No se pudo determinar la actividad para: {video_file.name}")
                continue
            
            if activity not in ACTIVITIES:
                print(f"Actividad desconocida: {activity}")
                continue
            
            # Procesar video
            features_list = self.data_processor.process_video(str(video_file), activity)
            
            # Añadir características y etiquetas
            for features in features_list:
                all_features.append(features)
                all_labels.append(activity)
        
        if not all_features:
            raise ValueError("No se pudieron extraer características de ningún video")
        
        print(f"Total de muestras extraídas: {len(all_features)}")
        
        # Contar muestras por clase
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("Distribución de clases:")
        for activity, count in label_counts.items():
            print(f"  {activity}: {count} muestras")
        
        return np.array(all_features), np.array(all_labels)
    
    def train_model(self, model_type: str, X: np.ndarray, y: np.ndarray,
                   save_path: Optional[str] = None) -> Dict:
        """
        Entrena un modelo específico
        
        Args:
            model_type: Tipo de modelo ('svm', 'rf', 'xgb')
            X: Características de entrenamiento
            y: Etiquetas de entrenamiento
            save_path: Ruta para guardar el modelo
            
        Returns:
            Diccionario con métricas del modelo
        """
        if model_type not in self.models:
            raise ValueError(f"Modelo {model_type} no disponible. Opciones: {list(self.models.keys())}")
        
        print(f"\n=== Entrenando modelo {model_type.upper()} ===")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL_CONFIG['test_size'], 
            random_state=MODEL_CONFIG['random_state'],
            stratify=y
        )
        
        # Normalizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Codificar etiquetas
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Entrenar modelo
        model = self.models[model_type]
        model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluar modelo
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Validación cruzada
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train_encoded, 
            cv=MODEL_CONFIG['cv_folds'], scoring='accuracy'
        )
        
        # Reporte detallado
        class_report = classification_report(
            y_test_encoded, y_pred, 
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test_encoded, y_pred)
        
        # Métricas del modelo
        metrics = {
            'model_type': model_type,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'classes': label_encoder.classes_.tolist(),
            'created_at': datetime.now().isoformat()
        }
        
        print(f"Precisión en test: {accuracy:.4f}")
        print(f"Validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print("\nReporte de clasificación:")
        print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
        
        # Guardar modelo si se especifica ruta
        if save_path:
            model_data = {
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'info': metrics
            }
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(model_data, save_path)
            print(f"\nModelo guardado en: {save_path}")
            
            # Guardar métricas en JSON
            metrics_path = save_path.with_suffix('.json')
            with open(metrics_path, 'w') as f:
                # Convertir arrays numpy a listas para JSON
                metrics_json = metrics.copy()
                if 'confusion_matrix' in metrics_json:
                    metrics_json['confusion_matrix'] = conf_matrix.tolist()
                json.dump(metrics_json, f, indent=2)
        
        return metrics
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        Entrena todos los modelos disponibles
        
        Args:
            X: Características de entrenamiento
            y: Etiquetas de entrenamiento
            
        Returns:
            Diccionario con métricas de todos los modelos
        """
        all_metrics = {}
        
        for model_type in self.models.keys():
            try:
                # Crear nombre de archivo único
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = MODELS_PATH / f"{model_type}_model_{timestamp}.pkl"
                
                # Entrenar modelo
                metrics = self.train_model(model_type, X, y, str(save_path))
                all_metrics[model_type] = metrics
                
            except Exception as e:
                print(f"Error entrenando modelo {model_type}: {e}")
                all_metrics[model_type] = {'error': str(e)}
        
        # Mostrar comparación de modelos
        print("\n=== COMPARACIÓN DE MODELOS ===")
        print("Modelo\t\tPrecisión\tCV Mean\t\tCV Std")
        print("-" * 50)
        
        for model_type, metrics in all_metrics.items():
            if 'error' not in metrics:
                print(f"{model_type.upper()}\t\t{metrics['accuracy']:.4f}\t\t"
                      f"{metrics['cv_mean']:.4f}\t\t{metrics['cv_std']:.4f}")
            else:
                print(f"{model_type.upper()}\t\tERROR: {metrics['error']}")
        
        return all_metrics
    
    def save_training_data(self, X: np.ndarray, y: np.ndarray, 
                          filename: str = None) -> str:
        """
        Guarda datos de entrenamiento procesados
        
        Args:
            X: Características
            y: Etiquetas
            filename: Nombre del archivo (opcional)
            
        Returns:
            Ruta del archivo guardado
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.pkl"
        
        save_path = DATA_PATH / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'features': X,
            'labels': y,
            'feature_names': self.data_processor.feature_extractor.get_feature_names(),
            'activities': list(ACTIVITIES.keys()),
            'created_at': datetime.now().isoformat(),
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
        joblib.dump(data, save_path)
        print(f"Datos de entrenamiento guardados en: {save_path}")
        
        return str(save_path)
    
    def load_training_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga datos de entrenamiento previamente guardados
        
        Args:
            filename: Nombre del archivo de datos
            
        Returns:
            Tupla con (características, etiquetas)
        """
        data_path = DATA_PATH / filename
        
        if not data_path.exists():
            raise FileNotFoundError(f"Archivo {data_path} no encontrado")
        
        data = joblib.load(data_path)
        
        print(f"Datos cargados: {data['n_samples']} muestras, {data['n_features']} características")
        print(f"Creado: {data.get('created_at', 'Desconocido')}")
        
        return data['features'], data['labels']


def main():
    """Función principal para entrenar modelos"""
    trainer = ModelTrainer()
    
    print("=== SISTEMA DE ENTRENAMIENTO POSETRACK AI ===\n")
    
    try:
        # Preparar datos de entrenamiento
        print("1. Extrayendo características de videos...")
        X, y = trainer.prepare_training_data()
        
        if len(X) == 0:
            print("Error: No se pudieron extraer características")
            return
        
        # Guardar datos procesados
        print("\n2. Guardando datos procesados...")
        trainer.save_training_data(X, y)
        
        # Entrenar todos los modelos
        print("\n3. Entrenando modelos...")
        all_metrics = trainer.train_all_models(X, y)
        
        # Mostrar resumen final
        print("\n=== ENTRENAMIENTO COMPLETADO ===")
        print(f"Total de muestras: {len(X)}")
        print(f"Características por muestra: {X.shape[1]}")
        print(f"Actividades: {len(set(y))}")
        
        # Encontrar mejor modelo
        best_model = None
        best_accuracy = 0
        
        for model_type, metrics in all_metrics.items():
            if 'error' not in metrics and metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model = model_type
        
        if best_model:
            print(f"\nMejor modelo: {best_model.upper()} (Precisión: {best_accuracy:.4f})")
        
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
