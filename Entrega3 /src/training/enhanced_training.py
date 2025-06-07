"""
Sistema de entrenamiento avanzado para PoseTrack AI - Entrega 3
Incluye optimización de hiperparámetros, ensemble, aumento de datos y características avanzadas
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import re
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
import warnings

warnings.filterwarnings("ignore")

from config.settings import (
    VIDEOS_PATH,
    MODELS_PATH,
    DATA_PATH,
    MODEL_CONFIG,
    ACTIVITIES,
)
from src.core.pose_tracker import PoseTracker
from src.core.feature_extractor import FeatureExtractor
from src.core.advanced_features import AdvancedFeatureExtractor, TemporalFilter
from src.core.data_augmentation import PoseAugmentation, ClassBalancer
from src.core.hyperparameter_optimization import HyperparameterOptimizer, AutoMLPipeline
from src.core.ensemble_models import EnsemblePredictor


class EnhancedVideoDataProcessor:
    """Procesador avanzado de videos para extraer datos de entrenamiento"""

    def __init__(self):
        self.pose_tracker = PoseTracker()
        self.feature_extractor = FeatureExtractor()
        self.advanced_feature_extractor = AdvancedFeatureExtractor()
        self.data_augmentation = PoseAugmentation()

    def process_video_advanced(
        self, video_path: str, activity_label: str, apply_augmentation: bool = True
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Procesa un video con extracción avanzada de características y aumento de datos

        Args:
            video_path: Ruta al video
            activity_label: Etiqueta de la actividad
            apply_augmentation: Si aplicar aumento de datos

        Returns:
            Tupla con (lista de características, lista de etiquetas)
        """
        features_list = []
        labels_list = []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return features_list, labels_list

        frame_count = 0
        self.feature_extractor.reset_buffer()

        print(f"Procesando: {Path(video_path).name}")

        # Almacenar secuencias de landmarks para análisis temporal
        landmarks_sequence = []
        angles_sequence = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar frame con MediaPipe
            annotated_frame, landmarks = self.pose_tracker.process_frame(frame)

            if landmarks:
                # Calcular ángulos
                angles = self.pose_tracker.calculate_angles(landmarks)

                # Almacenar para análisis temporal
                landmarks_sequence.append(np.array(landmarks, dtype=np.float32))
                # Convertir diccionario de ángulos a lista de valores
                angle_values = (
                    list(angles.values()) if isinstance(angles, dict) else angles
                )
                angles_sequence.append(np.array(angle_values, dtype=np.float32))

                # Añadir al buffer del extractor básico
                self.feature_extractor.add_frame_data(landmarks, angles, frame_count)

                # Extraer características básicas si el buffer está lleno
                if self.feature_extractor.is_ready():
                    basic_features = self.feature_extractor.extract_features()
                    if basic_features is not None:
                        # Extraer características avanzadas
                        advanced_features = (
                            self.advanced_feature_extractor.extract_temporal_features(
                                landmarks_sequence[-30:],  # Últimos 30 frames
                            )
                        )

                        # Combinar características básicas y avanzadas
                        combined_features = np.concatenate(
                            [basic_features, advanced_features]
                        )
                        features_list.append(combined_features)
                        labels_list.append(activity_label)

            frame_count += 1

        cap.release()

        # Aplicar aumento de datos si está habilitado
        if apply_augmentation and features_list:
            augmented_features, augmented_labels = (
                self.data_augmentation.augment_dataset(
                    np.array(features_list),
                    np.array(labels_list),
                    augmentation_factor=2,
                    balance_classes=True,
                )
            )

            # Combinar datos originales y aumentados
            all_features = np.vstack([features_list, augmented_features])
            all_labels = np.concatenate([labels_list, augmented_labels])

            print(
                f"Original: {len(features_list)}, Aumentado: {len(augmented_features)}, Total: {len(all_features)}"
            )

            return all_features.tolist(), all_labels.tolist()

        print(f"Extraídas {len(features_list)} secuencias de características")
        return features_list, labels_list

    def extract_activity_from_filename(self, filename: str) -> Optional[str]:
        """Extrae la etiqueta de actividad del nombre del archivo"""
        filename = filename.lower()

        patterns = {
            "acercarse": r"acer.*",
            "alejarse": r"alej.*",
            "girarD": r"girar.*d.*",
            "girarI": r"girar.*i.*",
            "sentarse": r"sent.*",
            "levantarse": r"levant.*",
        }

        for activity, pattern in patterns.items():
            if re.search(pattern, filename):
                return activity

        return None


class EnhancedModelTrainer:
    """Entrenador avanzado de modelos de clasificación de actividades"""

    def __init__(self):
        self.data_processor = EnhancedVideoDataProcessor()
        self.class_balancer = ClassBalancer()
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            n_trials=1,  # Reducido para acelerar
            cv_folds=3,  # Reducido para acelerar
            n_jobs=1,  # Serial para evitar problemas de memoria
        )
        self.automl_pipeline = AutoMLPipeline(self.hyperparameter_optimizer)
        self.ensemble_predictor = EnsemblePredictor()
        self.temporal_filter = TemporalFilter()

        self.scaler = None
        self.label_encoder = None

    def prepare_enhanced_training_data(
        self,
        videos_dir: str = None,
        apply_augmentation: bool = True,
        balance_classes: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara datos de entrenamiento mejorados con aumento de datos y balanceo de clases

        Args:
            videos_dir: Directorio con videos de entrenamiento
            apply_augmentation: Si aplicar aumento de datos
            balance_classes: Si balancear las clases

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

        # Buscar videos recursivamente
        video_extensions = [".mp4", ".avi", ".mov", ".MOV"]
        video_files = []

        for ext in video_extensions:
            # Buscar recursivamente en subdirectorios también
            video_files.extend(videos_dir.glob(f"*{ext}"))  # Root directory

        if not video_files:
            raise ValueError(f"No se encontraron videos en {videos_dir}")

        # Debug: mostrar videos encontrados y sus actividades detectadas
        print(f"Encontrados {len(video_files)} videos para procesar")
        activity_counts = {}

        for video_file in video_files:
            activity = self.data_processor.extract_activity_from_filename(
                video_file.name
            )
            if activity:
                activity_counts[activity] = activity_counts.get(activity, 0) + 1

        print(f"Distribución de actividades detectadas: {activity_counts}")

        for video_file in tqdm(video_files, desc="Procesando videos con mejoras"):
            # Extraer etiqueta del nombre del archivo
            activity = self.data_processor.extract_activity_from_filename(
                video_file.name
            )

            if activity is None:
                print(f"No se pudo determinar la actividad para: {video_file.name}")
                continue

            print(f"Video: {video_file.name} -> Actividad: {activity}")

            if activity not in ACTIVITIES:
                print(f"Actividad desconocida: {activity}")
                continue

            # Procesar video con características avanzadas
            features_list, labels_list = self.data_processor.process_video_advanced(
                str(video_file), activity, apply_augmentation
            )

            # Añadir características y etiquetas
            all_features.extend(features_list)
            all_labels.extend(labels_list)

        if not all_features:
            raise ValueError("No se pudieron extraer características de ningún video")

        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        print(f"Total de muestras extraídas: {len(all_features)}")

        # Debug: Mostrar clases únicas antes del balanceo
        unique_labels = np.unique(all_labels)
        print(f"Clases únicas detectadas: {unique_labels}")
        print(f"Número de clases: {len(unique_labels)}")

        # Mostrar distribución por clase
        for label in unique_labels:
            count = np.sum(np.array(all_labels) == label)
            print(f"  {label}: {count} muestras")

        # Balancear clases si está habilitado
        if balance_classes and len(unique_labels) > 1:
            print("Balanceando clases...")
            all_features, all_labels = ClassBalancer.oversample_minority_classes(
                all_features, all_labels, method="smote"
            )
            print(f"Muestras después del balanceo: {len(all_features)}")

        # Mostrar distribución de clases
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        print("Distribución final de clases:")
        for activity, count in label_counts.items():
            print(f"  {activity}: {count} muestras")

        return all_features, all_labels

    def train_optimized_models(
        self, X: np.ndarray, y: np.ndarray, optimization_trials: int = 100
    ) -> Dict[str, Dict]:
        """
        Entrena modelos con optimización de hiperparámetros

        Args:
            X: Características de entrenamiento
            y: Etiquetas de entrenamiento
            optimization_trials: Número de pruebas para optimización

        Returns:
            Diccionario con métricas y modelos optimizados
        """
        print(f"\n=== ENTRENAMIENTO CON OPTIMIZACIÓN DE HIPERPARÁMETROS ===")
        print(f"Muestras: {len(X)}, Características: {X.shape[1]}")

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Normalizar características
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Optimizar hiperparámetros para cada modelo
        optimized_models = {}
        model_metrics = {}

        model_types = ["rf", "xgb", "svm", "lightgbm", "mlp"]

        # Map model types to optimization names
        model_mapping = {
            "rf": "random_forest",
            "xgb": "xgboost",
            "svm": "svm",
            "lightgbm": "lightgbm",
            "mlp": "mlp",
        }

        for model_type in model_types:
            print(f"\nOptimizando {model_type.upper()}...")

            try:
                # Get the optimization name
                opt_name = model_mapping.get(model_type, model_type)

                # Optimize all models and get results for this specific model
                all_results = self.hyperparameter_optimizer.optimize_all_models(
                    X_train_scaled, y_train_encoded, models_to_optimize=[opt_name]
                )

                if opt_name not in all_results:
                    print(f"No se encontraron resultados para {model_type}")
                    continue

                model_result = all_results[opt_name]
                best_params = model_result["best_params"]
                best_score = model_result["best_score"]

                print(f"Mejores parámetros para {model_type}: {best_params}")
                print(f"Mejor puntuación CV: {best_score:.4f}")

                # Entrenar modelo con mejores parámetros
                if model_type == "rf":
                    from sklearn.ensemble import RandomForestClassifier

                    model = RandomForestClassifier(**best_params, random_state=42)
                elif model_type == "xgb":
                    import xgboost as xgb

                    model = xgb.XGBClassifier(**best_params, random_state=42)
                elif model_type == "svm":
                    from sklearn.svm import SVC

                    model = SVC(**best_params, random_state=42)
                elif model_type == "lightgbm":
                    import lightgbm as lgb

                    model = lgb.LGBMClassifier(**best_params, random_state=42)
                elif model_type == "mlp":
                    from sklearn.neural_network import MLPClassifier

                    model = MLPClassifier(**best_params, random_state=42)

                model.fit(X_train_scaled, y_train_encoded)

                # Evaluar modelo
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = (
                    model.predict_proba(X_test_scaled)
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Calcular métricas
                accuracy = accuracy_score(y_test_encoded, y_pred)
                f1 = f1_score(y_test_encoded, y_pred, average="weighted")
                precision, recall, _, _ = precision_recall_fscore_support(
                    y_test_encoded, y_pred, average="weighted"
                )

                optimized_models[model_type] = model
                model_metrics[model_type] = {
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                    "best_params": best_params,
                    "cv_score": best_score,
                }

                print(f"Resultados {model_type}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            except Exception as e:
                print(f"Error optimizando {model_type}: {e}")
                model_metrics[model_type] = {"error": str(e)}

        return {
            "models": optimized_models,
            "metrics": model_metrics,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "test_data": (X_test_scaled, y_test_encoded),
            "X_train": X_train_scaled,
            "y_train": y_train_encoded,
        }

    def create_ensemble_model(self, optimized_results: Dict) -> Dict:
        """
        Crea y entrena modelos de ensemble

        Args:
            optimized_results: Resultados de modelos optimizados

        Returns:
            Diccionario con ensemble y métricas
        """
        print(f"\n=== CREANDO MODELOS DE ENSEMBLE ===")

        models = optimized_results["models"]
        X_test, y_test = optimized_results["test_data"]

        if len(models) < 2:
            print("Se necesitan al menos 2 modelos para crear ensemble")
            return {}

        # Establecer los modelos base en el ensemble predictor
        self.ensemble_predictor.base_models = models

        ensemble_results = {}

        try:
            # Crear voting ensemble
            print("🗳️ Creando Voting Ensemble...")
            voting_ensemble = self.ensemble_predictor.create_voting_ensemble(
                voting_type="soft"
            )

            # Necesitamos los datos de entrenamiento
            X_train = optimized_results.get("X_train")
            y_train = optimized_results.get("y_train")

            if X_train is not None and y_train is not None:
                voting_ensemble.fit(X_train, y_train)
                voting_pred = voting_ensemble.predict(X_test)
                voting_accuracy = accuracy_score(y_test, voting_pred)

                ensemble_results["voting"] = {
                    "model": voting_ensemble,
                    "accuracy": voting_accuracy,
                    "predictions": voting_pred,
                }
                print(f"✅ Voting Ensemble - Precisión: {voting_accuracy:.4f}")

                # Crear stacking ensemble
                print("🏗️ Creando Stacking Ensemble...")
                stacking_ensemble = self.ensemble_predictor.create_stacking_ensemble()
                stacking_ensemble.fit(X_train, y_train)
                stacking_pred = stacking_ensemble.predict(X_test)
                stacking_accuracy = accuracy_score(y_test, stacking_pred)

                ensemble_results["stacking"] = {
                    "model": stacking_ensemble,
                    "accuracy": stacking_accuracy,
                    "predictions": stacking_pred,
                }
                print(f"✅ Stacking Ensemble - Precisión: {stacking_accuracy:.4f}")
            else:
                print("❌ Datos de entrenamiento no disponibles para ensemble")

        except Exception as e:
            print(f"❌ Error creando ensemble: {str(e)}")
            return {}

        # Mostrar resumen
        if ensemble_results:
            print("\n📊 RESUMEN DE ENSEMBLE:")
            print("=" * 40)
            for strategy, result in ensemble_results.items():
                print(f"{strategy.upper()}: {result['accuracy']:.4f}")

        return ensemble_results

    def train_automl_pipeline(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Entrena usando pipeline AutoML

        Args:
            X: Características de entrenamiento
            y: Etiquetas de entrenamiento

        Returns:
            Diccionario con resultados AutoML
        """
        print(f"\n=== ENTRENAMIENTO AUTOML ===")

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Entrenar con AutoML
        automl_results = self.automl_pipeline.auto_train_and_optimize(
            X_train, y_train, X_test, y_test
        )

        return automl_results

    def save_enhanced_models(
        self,
        training_results: Dict,
        ensemble_results: Dict,
        automl_results: Dict,
        timestamp: str = None,
    ) -> Dict[str, str]:
        """
        Guarda todos los modelos entrenados

        Args:
            training_results: Resultados del entrenamiento optimizado
            ensemble_results: Resultados de ensemble
            automl_results: Resultados AutoML
            timestamp: Timestamp para nombres de archivo

        Returns:
            Diccionario con rutas de archivos guardados
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = {}

        # Crear directorio de modelos si no existe
        MODELS_PATH.mkdir(exist_ok=True)

        # Guardar modelos optimizados
        for model_name, model in training_results["models"].items():
            model_path = MODELS_PATH / f"{model_name}_optimized_{timestamp}.pkl"
            joblib.dump(model, model_path)
            saved_files[f"{model_name}_optimized"] = str(model_path)

        # Guardar scaler y label encoder
        scaler_path = MODELS_PATH / f"scaler_{timestamp}.pkl"
        encoder_path = MODELS_PATH / f"label_encoder_{timestamp}.pkl"

        joblib.dump(training_results["scaler"], scaler_path)
        joblib.dump(training_results["label_encoder"], encoder_path)

        saved_files["scaler"] = str(scaler_path)
        saved_files["label_encoder"] = str(encoder_path)

        # Guardar configuración de ensemble
        if ensemble_results:
            ensemble_config_path = MODELS_PATH / f"ensemble_config_{timestamp}.json"
            ensemble_config = {
                "ensemble_results": ensemble_results,
                "model_files": {
                    name: str(path)
                    for name, path in saved_files.items()
                    if "optimized" in name
                },
            }

            with open(ensemble_config_path, "w") as f:
                json.dump(ensemble_config, f, indent=2, default=str)

            saved_files["ensemble_config"] = str(ensemble_config_path)

        # Guardar resultados completos
        results_path = MODELS_PATH / f"training_results_{timestamp}.json"
        complete_results = {
            "training_metrics": training_results["metrics"],
            "ensemble_metrics": ensemble_results,
            "automl_metrics": automl_results,
            "timestamp": timestamp,
            "saved_files": saved_files,
        }

        with open(results_path, "w") as f:
            json.dump(complete_results, f, indent=2, default=str)

        saved_files["training_results"] = str(results_path)

        return saved_files

    def save_training_data(
        self, X: np.ndarray, y: np.ndarray, filename: str = None
    ) -> str:
        """Guarda datos de entrenamiento procesados"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_training_data_{timestamp}.pkl"

        data_path = DATA_PATH / filename
        DATA_PATH.mkdir(exist_ok=True)

        training_data = {
            "features": X,
            "labels": y,
            "n_samples": len(X),
            "n_features": X.shape[1] if len(X) > 0 else 0,
            "n_classes": len(set(y)) if len(y) > 0 else 0,
            "created_at": datetime.now().isoformat(),
            "activities": list(set(y)) if len(y) > 0 else [],
        }

        joblib.dump(training_data, data_path)
        print(f"Datos guardados en: {data_path}")

        return str(data_path)


def main():
    """Función principal para entrenamiento mejorado"""
    trainer = EnhancedModelTrainer()

    print("=== POSETRACK AI - ENTRENAMIENTO AVANZADO (ENTREGA 3) ===\n")

    try:
        # 1. Preparar datos con mejoras
        print("1. Extrayendo características avanzadas con aumento de datos...")
        X, y = trainer.prepare_enhanced_training_data(
            apply_augmentation=True, balance_classes=True
        )

        if len(X) == 0:
            print("Error: No se pudieron extraer características")
            return

        # 2. Guardar datos procesados
        print("\n2. Guardando datos procesados...")
        data_path = trainer.save_training_data(X, y)

        # 3. Entrenar modelos con optimización
        print("\n3. Entrenando modelos con optimización de hiperparámetros...")
        training_results = trainer.train_optimized_models(X, y, optimization_trials=50)

        # 4. Crear modelos de ensemble
        print("\n4. Creando modelos de ensemble...")
        ensemble_results = trainer.create_ensemble_model(training_results)

        # 5. Entrenar con AutoML
        print("\n5. Entrenando con AutoML...")
        automl_results = trainer.train_automl_pipeline(X, y)

        # 6. Guardar todos los modelos
        print("\n6. Guardando modelos...")
        saved_files = trainer.save_enhanced_models(
            training_results, ensemble_results, automl_results
        )

        # 7. Mostrar resumen final
        print("\n=== ENTRENAMIENTO AVANZADO COMPLETADO ===")
        print(f"Total de muestras: {len(X)}")
        print(f"Características por muestra: {X.shape[1]}")
        print(f"Actividades: {len(set(y))}")
        print(f"Datos guardados en: {data_path}")

        # Encontrar mejores modelos
        print("\n=== RESULTADOS DE MODELOS OPTIMIZADOS ===")
        best_individual = None
        best_individual_score = 0

        for model_type, metrics in training_results["metrics"].items():
            if "accuracy" in metrics:
                accuracy = metrics["accuracy"]
                f1 = metrics["f1_score"]
                print(f"{model_type.upper()}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

                if accuracy > best_individual_score:
                    best_individual_score = accuracy
                    best_individual = model_type

        if best_individual:
            print(
                f"\nMejor modelo individual: {best_individual.upper()} (Accuracy: {best_individual_score:.4f})"
            )

        # Resultados de ensemble
        print("\n=== RESULTADOS DE ENSEMBLE ===")
        best_ensemble = None
        best_ensemble_score = 0

        for strategy, metrics in ensemble_results.items():
            if "accuracy" in metrics:
                accuracy = metrics["accuracy"]
                f1 = metrics["f1_score"]
                print(f"Ensemble {strategy}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

                if accuracy > best_ensemble_score:
                    best_ensemble_score = accuracy
                    best_ensemble = strategy

        if best_ensemble:
            print(
                f"\nMejor ensemble: {best_ensemble} (Accuracy: {best_ensemble_score:.4f})"
            )

        # Resultados AutoML
        if automl_results and "best_model" in automl_results:
            print(f"\n=== RESULTADOS AUTOML ===")
            print(f"Mejor modelo AutoML: {automl_results['best_model']}")
            print(f"Accuracy: {automl_results['best_score']:.4f}")

        print(f"\nArchivos guardados:")
        for name, path in saved_files.items():
            print(f"  {name}: {path}")

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
