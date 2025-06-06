"""
Specialized model training system for PoseTrack AI

This module provides functionality to train specialized models for different
activity categories (basic activities and gym exercises). It implements a
hierarchical approach to activity recognition by training separate models
for different types of activities.
"""

import cv2
import pandas as pd
from pathlib import Path
import joblib
import numpy as np
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

from config.settings import (
    VIDEOS_PATH, MODELS_PATH, DATA_PATH, MODEL_CONFIG, 
    BASIC_ACTIVITIES, GYM_ACTIVITIES, ACTIVITIES
)
from src.core.pose_tracker import PoseTracker
from src.core.feature_extractor import FeatureExtractor


class SpecializedVideoDataProcessor:
    """Process videos to extract training data for specialized models
    
    This class handles video processing and feature extraction for training
    specialized machine learning models. It includes methods to process
    video files, detect poses, and extract relevant features for different
    activity categories.
    """
    
    def __init__(self):
        self.pose_tracker = PoseTracker()
        self.feature_extractor = FeatureExtractor()
    def process_video(self, video_path: str, activity_label: str) -> List[np.ndarray]:
        """
        Process a video and extract features
        
        This method processes a video file frame by frame to detect human poses,
        track landmarks, and extract relevant features for activity recognition.
        It accumulates features across the video timeline to create training
        samples.
        
        Args:
            video_path: Path to the video file
            activity_label: Activity label for the video
            
        Returns:
            List of feature arrays extracted from the video
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
            'acercarse': r'acerca.*',
            'alejarse': r'aleja.*',
            'girarD': r'girar.*d.*',
            'girarI': r'girar.*i.*',
            'sentarse': r'sent.*',
            'levantarse': r'levant.*',
            'parado': r'parado.*',
            'squat': r'squat.*',
            'russian_twist': r'russian.*twist.*',
            'push_up': r'push.*up.*'
        }
        
        for activity, pattern in patterns.items():
            if re.search(pattern, filename):
                return activity
        
        return None


class SpecializedModelTrainer:
    """Entrenador de modelos especializados para actividades básicas y de gimnasio"""
    
    def __init__(self):
        self.data_processor = SpecializedVideoDataProcessor()
        self.models = {
            'svm': SVC(**MODEL_CONFIG['models']['svm'], probability=True),
            'rf': RandomForestClassifier(**MODEL_CONFIG['models']['rf']),
            'xgb': xgb.XGBClassifier(**MODEL_CONFIG['models']['xgb'])
        }
        
    def prepare_training_data_by_category(self, videos_dir: str = None, 
                                        activity_set: Dict[str, str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos de entrenamiento para un conjunto específico de actividades
        
        Args:
            videos_dir: Directorio con videos de entrenamiento
            activity_set: Conjunto de actividades a incluir (BASIC_ACTIVITIES o GYM_ACTIVITIES)
            
        Returns:
            Tupla con (características, etiquetas)
        """
        if videos_dir is None:
            videos_dir = VIDEOS_PATH
        
        if activity_set is None:
            activity_set = ACTIVITIES
        
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
        print(f"Actividades objetivo: {list(activity_set.keys())}")
        
        processed_count = 0
        for video_file in tqdm(video_files, desc="Procesando videos"):
            # Extraer etiqueta del nombre del archivo
            activity = self.data_processor.extract_activity_from_filename(video_file.name)
            
            if activity is None:
                continue
            
            # Filtrar solo las actividades del conjunto especificado
            if activity not in activity_set:
                continue
            
            processed_count += 1
            print(f"Procesando {activity}: {video_file.name}")
            
            # Procesar video
            features_list = self.data_processor.process_video(str(video_file), activity)
            
            # Añadir características y etiquetas
            for features in features_list:
                all_features.append(features)
                all_labels.append(activity)
        
        if not all_features:
            raise ValueError(f"No se pudieron extraer características para las actividades: {list(activity_set.keys())}")
        
        print(f"Videos procesados: {processed_count}")
        print(f"Total de muestras extraídas: {len(all_features)}")
        
        # Contar muestras por clase
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("Distribución de clases:")
        for activity, count in label_counts.items():
            print(f"  {activity}: {count} muestras")
        
        return np.array(all_features), np.array(all_labels)
    
    def train_specialized_model(self, model_type: str, X: np.ndarray, y: np.ndarray,
                              model_category: str, save_path: Optional[str] = None) -> Dict:
        """
        Entrena un modelo especializado
        
        Args:
            model_type: Tipo de modelo ('svm', 'rf', 'xgb')
            X: Características de entrenamiento
            y: Etiquetas de entrenamiento
            model_category: Categoría del modelo ('basic' o 'gym')
            save_path: Ruta para guardar el modelo
            
        Returns:
            Diccionario con métricas del modelo
        """
        if model_type not in self.models:
            raise ValueError(f"Modelo {model_type} no disponible. Opciones: {list(self.models.keys())}")
        
        print(f"\n=== Entrenando modelo {model_type.upper()} - {model_category.upper()} ===")
        
        # Verificar que tenemos suficientes datos
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            raise ValueError(f"Necesitas al menos 2 clases diferentes para entrenar un modelo. Encontradas: {unique_labels}")
        
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
            cv=min(MODEL_CONFIG['cv_folds'], len(unique_labels)), 
            scoring='accuracy'
        )
        
        # Reporte detallado
        target_names = label_encoder.classes_
        class_report = classification_report(
            y_test_encoded, y_pred, 
            target_names=target_names, 
            output_dict=True
        )
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test_encoded, y_pred)
        
        # Métricas del modelo
        metrics = {
            'model_type': model_type,
            'category': model_category,
            'accuracy': float(accuracy),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'classes': target_names.tolist(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'created_at': datetime.now().isoformat()
        }
        
        # Mostrar resultados
        print(f"Precisión: {accuracy:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nReporte de clasificación:")
        print(classification_report(y_test_encoded, y_pred, target_names=target_names))
        
        # Guardar modelo
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = MODELS_PATH / f"{model_type}_{model_category}_model_{timestamp}.pkl"
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'info': metrics
        }
        
        joblib.dump(model_data, save_path)
        print(f"Modelo guardado en: {save_path}")
        
        # Guardar métricas por separado
        metrics_path = save_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def train_all_specialized_models(self):
        """
        Entrena todos los modelos especializados (básicos y de gimnasio)
        """
        print("🎯 Iniciando entrenamiento de modelos especializados")
        print("=" * 60)
        
        all_results = {}
        
        # 1. Entrenar modelos para actividades básicas
        print("\n🏃 ENTRENANDO MODELOS PARA ACTIVIDADES BÁSICAS")
        print("=" * 50)
        
        try:
            X_basic, y_basic = self.prepare_training_data_by_category(
                activity_set=BASIC_ACTIVITIES
            )
            
            basic_results = {}
            for model_type in ['svm', 'rf', 'xgb']:
                try:
                    metrics = self.train_specialized_model(
                        model_type, X_basic, y_basic, 'basic'
                    )
                    basic_results[model_type] = metrics
                except Exception as e:
                    print(f"Error entrenando modelo básico {model_type}: {e}")
                    basic_results[model_type] = {'error': str(e)}
            
            all_results['basic'] = basic_results
            
        except Exception as e:
            print(f"Error en entrenamiento de modelos básicos: {e}")
            all_results['basic'] = {'error': str(e)}
        
        # 2. Entrenar modelos para actividades de gimnasio
        print("\n🏋️ ENTRENANDO MODELOS PARA ACTIVIDADES DE GIMNASIO")
        print("=" * 50)
        
        try:
            X_gym, y_gym = self.prepare_training_data_by_category(
                activity_set=GYM_ACTIVITIES
            )
            
            gym_results = {}
            for model_type in ['svm', 'rf', 'xgb']:
                try:
                    metrics = self.train_specialized_model(
                        model_type, X_gym, y_gym, 'gym'
                    )
                    gym_results[model_type] = metrics
                except Exception as e:
                    print(f"Error entrenando modelo gimnasio {model_type}: {e}")
                    gym_results[model_type] = {'error': str(e)}
            
            all_results['gym'] = gym_results
            
        except Exception as e:
            print(f"Error en entrenamiento de modelos de gimnasio: {e}")
            all_results['gym'] = {'error': str(e)}
        
        # 3. Mostrar resumen final
        self.show_specialized_results_summary(all_results)
        
        return all_results
    
    def show_specialized_results_summary(self, results: Dict):
        """
        Muestra un resumen de los resultados de entrenamiento
        """
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE MODELOS ESPECIALIZADOS")
        print("=" * 60)
        
        for category in ['basic', 'gym']:
            category_name = "ACTIVIDADES BÁSICAS" if category == 'basic' else "EJERCICIOS DE GIMNASIO"
            print(f"\n🎯 {category_name}")
            print("-" * 50)
            
            if category in results and not isinstance(results[category].get('error'), str):
                print(f"{'Modelo':<10} {'Precisión':<12} {'CV Mean':<10} {'CV Std':<10}")
                print("-" * 45)
                
                best_model = None
                best_accuracy = 0
                
                for model_type, metrics in results[category].items():
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        accuracy = metrics['accuracy']
                        cv_mean = metrics['cv_mean']
                        cv_std = metrics['cv_std']
                        
                        print(f"{model_type.upper():<10} {accuracy:<12.4f} {cv_mean:<10.4f} {cv_std:<10.4f}")
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model_type
                
                if best_model:
                    print(f"\n✅ Mejor modelo {category_name}: {best_model.upper()} (Precisión: {best_accuracy:.4f})")
            else:
                print(f"❌ Error en entrenamiento: {results[category].get('error', 'Error desconocido')}")


def main():
    """Función principal para entrenar modelos especializados"""
    trainer = SpecializedModelTrainer()
    
    try:
        results = trainer.train_all_specialized_models()
        print("\n🎉 Entrenamiento de modelos especializados completado!")
        
    except Exception as e:
        print(f"❌ Error en el entrenamiento: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
