#!/usr/bin/env python3
"""
Entrenamiento rápido para PoseTrack AI - Entrega 3
Versión optimizada para ejecución rápida
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
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


class QuickTrainer:
    """Entrenador rápido con optimización básica"""

    def __init__(self):
        self.pose_tracker = PoseTracker()
        self.feature_extractor = FeatureExtractor()
        self.advanced_feature_extractor = AdvancedFeatureExtractor()
        self.data_augmentation = PoseAugmentation()
        self.class_balancer = ClassBalancer()
        self.temporal_filter = TemporalFilter()

        self.scaler = None
        self.label_encoder = None

    def process_video_advanced(
        self, video_path: str, activity_label: str, apply_augmentation: bool = True
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Procesa un video con extracción avanzada de características"""
        features_list = []
        labels_list = []

        try:
            cap = cv2.VideoCapture(video_path)
            poses_sequence = []

            print(f"Procesando video: {Path(video_path).name}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detectar pose
                results = self.pose_tracker.detect_pose(frame)
                if results.pose_landmarks:
                    # Convertir landmarks a array numpy
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                    poses_sequence.append(np.array(landmarks))

            cap.release()

            if len(poses_sequence) < 10:  # Mínimo 10 frames
                print(f"Video muy corto: {len(poses_sequence)} frames")
                return [], []

            # Extraer características avanzadas
            advanced_features = (
                self.advanced_feature_extractor.extract_temporal_features(
                    poses_sequence
                )
            )

            if advanced_features is not None and len(advanced_features) > 0:
                features_list.append(advanced_features)
                labels_list.append(activity_label)

                # Aplicar aumento de datos si está habilitado
                if apply_augmentation:
                    augmented_poses = self.data_augmentation.augment_pose_sequence(
                        poses_sequence
                    )

                    for aug_poses in augmented_poses:
                        if len(aug_poses) >= 10:
                            aug_features = self.advanced_feature_extractor.extract_temporal_features(
                                aug_poses
                            )
                            if aug_features is not None and len(aug_features) > 0:
                                features_list.append(aug_features)
                                labels_list.append(activity_label)

        except Exception as e:
            print(f"Error procesando video {video_path}: {str(e)}")

        return features_list, labels_list

    def prepare_training_data(
        self,
        videos_dir: str = None,
        apply_augmentation: bool = True,
        balance_classes: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepara datos de entrenamiento"""
        if videos_dir is None:
            videos_dir = str(VIDEOS_PATH)

        videos_dir = Path(videos_dir)
        all_features = []
        all_labels = []

        print(f"🎬 Procesando videos desde: {videos_dir}")

        # Procesar todos los videos
        for video_file in videos_dir.glob("*.mp4"):
            activity = self.extract_activity_from_filename(video_file.name)
            if activity and activity in ACTIVITIES:
                print(f"Video: {video_file.name} -> Actividad: {activity}")

                features_list, labels_list = self.process_video_advanced(
                    str(video_file), activity, apply_augmentation
                )

                all_features.extend(features_list)
                all_labels.extend(labels_list)

        # También procesar archivos .MOV
        for video_file in videos_dir.glob("*.MOV"):
            activity = self.extract_activity_from_filename(video_file.name)
            if activity and activity in ACTIVITIES:
                print(f"Video: {video_file.name} -> Actividad: {activity}")

                features_list, labels_list = self.process_video_advanced(
                    str(video_file), activity, apply_augmentation
                )

                all_features.extend(features_list)
                all_labels.extend(labels_list)

        if not all_features:
            raise ValueError("No se pudieron extraer características de ningún video")

        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        print(f"📊 Total de muestras extraídas: {len(all_features)}")

        # Mostrar distribución por clase
        unique_labels = np.unique(all_labels)
        print(f"Clases detectadas: {unique_labels}")
        for label in unique_labels:
            count = np.sum(all_labels == label)
            print(f"  {label}: {count} muestras")

        # Balancear clases si está habilitado
        if balance_classes and len(unique_labels) > 1:
            print("⚖️ Balanceando clases...")
            all_features, all_labels = ClassBalancer.oversample_minority_classes(
                all_features, all_labels, method="smote"
            )
            print(f"📊 Muestras después del balanceo: {len(all_features)}")

        return all_features, all_labels

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

    def train_quick_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Entrena modelos con configuración rápida"""
        print(f"\n🚀 ENTRENAMIENTO RÁPIDO DE MODELOS")
        print(f"Muestras: {len(X)}, Características: {X.shape[1]}")
        print("=" * 60)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Escalar datos
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        models = {}
        results = {}

        # Random Forest - configuración rápida
        print("🌲 Entrenando Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=50,  # Reducido para velocidad
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train_scaled, y_train_encoded)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test_encoded, rf_pred)

        models["random_forest"] = rf_model
        results["random_forest"] = {"accuracy": rf_accuracy, "predictions": rf_pred}
        print(f"✅ Random Forest - Precisión: {rf_accuracy:.4f}")

        # XGBoost - configuración rápida
        print("🚀 Entrenando XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,  # Reducido para velocidad
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        )
        xgb_model.fit(X_train_scaled, y_train_encoded)
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_accuracy = accuracy_score(y_test_encoded, xgb_pred)

        models["xgboost"] = xgb_model
        results["xgboost"] = {"accuracy": xgb_accuracy, "predictions": xgb_pred}
        print(f"✅ XGBoost - Precisión: {xgb_accuracy:.4f}")

        # SVM - configuración rápida
        print("🎯 Entrenando SVM...")
        svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        svm_model.fit(X_train_scaled, y_train_encoded)
        svm_pred = svm_model.predict(X_test_scaled)
        svm_accuracy = accuracy_score(y_test_encoded, svm_pred)

        models["svm"] = svm_model
        results["svm"] = {"accuracy": svm_accuracy, "predictions": svm_pred}
        print(f"✅ SVM - Precisión: {svm_accuracy:.4f}")

        # Crear ensemble simple por votación
        print("🏆 Creando ensemble por votación...")
        ensemble_pred = []
        for i in range(len(y_test_encoded)):
            votes = [rf_pred[i], xgb_pred[i], svm_pred[i]]
            # Votación mayoritaria
            ensemble_pred.append(max(set(votes), key=votes.count))

        ensemble_pred = np.array(ensemble_pred)
        ensemble_accuracy = accuracy_score(y_test_encoded, ensemble_pred)

        results["ensemble"] = {
            "accuracy": ensemble_accuracy,
            "predictions": ensemble_pred,
        }
        print(f"✅ Ensemble - Precisión: {ensemble_accuracy:.4f}")

        print("\n📊 RESUMEN DE RESULTADOS:")
        print("=" * 40)
        for model_name, result in results.items():
            print(f"{model_name.upper()}: {result['accuracy']:.4f}")

        # Guardar modelos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_info = {
            "timestamp": timestamp,
            "models": models,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "results": results,
            "test_data": (X_test_scaled, y_test_encoded),
        }

        model_path = MODELS_PATH / f"quick_models_{timestamp}.pkl"
        joblib.dump(models_info, model_path)
        print(f"💾 Modelos guardados en: {model_path}")

        return models_info


def main():
    """Función principal para entrenamiento rápido"""
    print("🚀 PoseTrack AI - Entrenamiento Rápido - Entrega 3")
    print("=" * 60)

    try:
        trainer = QuickTrainer()

        # 1. Preparar datos
        print("\n1. 📊 Preparando datos de entrenamiento...")
        X, y = trainer.prepare_training_data(
            apply_augmentation=True, balance_classes=True
        )

        # 2. Entrenar modelos
        print("\n2. 🎯 Entrenando modelos...")
        results = trainer.train_quick_models(X, y)

        print("\n✅ ¡Entrenamiento completado exitosamente!")
        print("Modelos entrenados:")
        for model_name, result in results["results"].items():
            print(f"  - {model_name.upper()}: {result['accuracy']:.4f}")

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
