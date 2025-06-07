"""
Módulo de modelos ensemble avanzados
Implementa voting, stacking, boosting y ensemble temporal
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from collections import defaultdict, deque
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class EnsemblePredictor:
    """Predictor ensemble avanzado con múltiples estrategias"""

    def __init__(self, random_state: int = 42):
        """
        Inicializa el predictor ensemble

        Args:
            random_state: Semilla aleatoria
        """
        self.random_state = random_state
        self.base_models = {}
        self.ensemble_models = {}
        self.model_weights = {}
        self.trained = False

    def create_base_models(
        self, optimized_params: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Crea modelos base para el ensemble

        Args:
            optimized_params: Parámetros optimizados para cada modelo

        Returns:
            Diccionario con modelos base
        """
        base_models = {}

        # Parámetros por defecto si no se proporcionan optimizados
        default_params = {
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.random_state,
            },
            "extra_trees": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.random_state,
            },
            "xgboost": {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "verbosity": 0,
            },
            "lightgbm": {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_state,
                "verbosity": -1,
            },
            "svm": {
                "kernel": "rbf",
                "C": 10,
                "gamma": "scale",
                "probability": True,
                "random_state": self.random_state,
            },
            "mlp": {
                "hidden_layer_sizes": (256, 128, 64),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.001,
                "max_iter": 1000,
                "random_state": self.random_state,
                "early_stopping": True,
            },
        }

        # Usar parámetros optimizados si están disponibles
        params = optimized_params if optimized_params else default_params

        # Random Forest
        if "random_forest" in params:
            base_models["random_forest"] = RandomForestClassifier(
                **params["random_forest"]
            )

        # Extra Trees
        if "extra_trees" in params:
            base_models["extra_trees"] = ExtraTreesClassifier(**params["extra_trees"])

        # XGBoost
        if "xgboost" in params:
            base_models["xgboost"] = xgb.XGBClassifier(**params["xgboost"])

        # LightGBM
        if "lightgbm" in params:
            base_models["lightgbm"] = lgb.LGBMClassifier(**params["lightgbm"])

        # SVM
        if "svm" in params:
            base_models["svm"] = SVC(**params["svm"])

        # MLP
        if "mlp" in params:
            base_models["mlp"] = MLPClassifier(**params["mlp"])

        self.base_models = base_models
        return base_models

    def create_voting_ensemble(self, voting_type: str = "soft") -> VotingClassifier:
        """
        Crea ensemble de votación

        Args:
            voting_type: Tipo de votación ('hard' o 'soft')

        Returns:
            Modelo de votación ensemble
        """
        if not self.base_models:
            raise ValueError("Debe crear modelos base primero")

        estimators = [(name, model) for name, model in self.base_models.items()]

        voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)

        self.ensemble_models["voting"] = voting_clf
        return voting_clf

    def create_stacking_ensemble(
        self, meta_learner=None, cv: int = 5
    ) -> StackingClassifier:
        """
        Crea ensemble de stacking

        Args:
            meta_learner: Modelo meta-aprendiz
            cv: Número de folds para stacking

        Returns:
            Modelo de stacking ensemble
        """
        if not self.base_models:
            raise ValueError("Debe crear modelos base primero")

        if meta_learner is None:
            meta_learner = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state
            )

        estimators = [(name, model) for name, model in self.base_models.items()]

        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv,
            stack_method="predict_proba",
        )

        self.ensemble_models["stacking"] = stacking_clf
        return stacking_clf

    def create_weighted_ensemble(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "WeightedEnsemble":
        """
        Crea ensemble ponderado basado en rendimiento de validación

        Args:
            X_val: Datos de validación
            y_val: Etiquetas de validación

        Returns:
            Modelo ensemble ponderado
        """
        if not self.base_models:
            raise ValueError("Debe crear modelos base primero")

        # Calcular pesos basados en precisión de validación
        weights = {}
        total_score = 0

        for name, model in self.base_models.items():
            if hasattr(model, "predict"):
                try:
                    score = model.score(X_val, y_val)
                    weights[name] = max(score, 0.01)  # Peso mínimo
                    total_score += weights[name]
                except:
                    weights[name] = 0.01

        # Normalizar pesos
        for name in weights:
            weights[name] /= total_score

        self.model_weights = weights

        weighted_ensemble = WeightedEnsemble(self.base_models, weights)
        self.ensemble_models["weighted"] = weighted_ensemble

        return weighted_ensemble

    def create_dynamic_ensemble(
        self, confidence_threshold: float = 0.7
    ) -> "DynamicEnsemble":
        """
        Crea ensemble dinámico que selecciona modelos por confianza

        Args:
            confidence_threshold: Umbral de confianza para selección

        Returns:
            Modelo ensemble dinámico
        """
        if not self.base_models:
            raise ValueError("Debe crear modelos base primero")

        dynamic_ensemble = DynamicEnsemble(self.base_models, confidence_threshold)
        self.ensemble_models["dynamic"] = dynamic_ensemble

        return dynamic_ensemble

    def train_all_ensembles(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Entrena todos los tipos de ensemble

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación
            y_val: Etiquetas de validación

        Returns:
            Diccionario con resultados de entrenamiento
        """
        results = {}

        print("🏋️ Entrenando modelos base...")
        # Entrenar modelos base
        for name, model in self.base_models.items():
            print(f"   Entrenando {name}...")
            model.fit(X_train, y_train)

            # Evaluar modelo base
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)

            results[name] = {
                "train_score": train_score,
                "val_score": val_score,
                "type": "base_model",
            }

        print("\n🎭 Entrenando modelos ensemble...")

        # Crear y entrenar ensemble de votación
        voting_clf = self.create_voting_ensemble("soft")
        voting_clf.fit(X_train, y_train)

        train_score = voting_clf.score(X_train, y_train)
        val_score = voting_clf.score(X_val, y_val)

        results["voting"] = {
            "train_score": train_score,
            "val_score": val_score,
            "type": "ensemble",
        }

        # Crear y entrenar ensemble de stacking
        stacking_clf = self.create_stacking_ensemble()
        stacking_clf.fit(X_train, y_train)

        train_score = stacking_clf.score(X_train, y_train)
        val_score = stacking_clf.score(X_val, y_val)

        results["stacking"] = {
            "train_score": train_score,
            "val_score": val_score,
            "type": "ensemble",
        }

        # Crear ensemble ponderado
        weighted_ensemble = self.create_weighted_ensemble(X_val, y_val)
        # El ensemble ponderado ya está "entrenado" al crear los pesos

        train_score = weighted_ensemble.score(X_train, y_train)
        val_score = weighted_ensemble.score(X_val, y_val)

        results["weighted"] = {
            "train_score": train_score,
            "val_score": val_score,
            "type": "ensemble",
            "weights": self.model_weights.copy(),
        }

        # Crear ensemble dinámico
        dynamic_ensemble = self.create_dynamic_ensemble()
        # El ensemble dinámico no requiere entrenamiento adicional

        train_score = dynamic_ensemble.score(X_train, y_train)
        val_score = dynamic_ensemble.score(X_val, y_val)

        results["dynamic"] = {
            "train_score": train_score,
            "val_score": val_score,
            "type": "ensemble",
        }

        self.trained = True

        return results

    def get_best_ensemble(self, results: Dict[str, Any]) -> Tuple[str, Any]:
        """
        Selecciona el mejor ensemble basado en puntuación de validación

        Args:
            results: Resultados de entrenamiento

        Returns:
            Tupla (nombre_mejor_modelo, modelo_mejor)
        """
        best_score = -1
        best_name = None

        for name, result in results.items():
            val_score = result["val_score"]
            if val_score > best_score:
                best_score = val_score
                best_name = name

        best_model = None
        if best_name in self.base_models:
            best_model = self.base_models[best_name]
        elif best_name in self.ensemble_models:
            best_model = self.ensemble_models[best_name]

        return best_name, best_model

    def save_models(self, save_path: str):
        """
        Guarda todos los modelos entrenados

        Args:
            save_path: Ruta donde guardar los modelos
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Guardar modelos base
        base_path = save_path / "base_models"
        base_path.mkdir(exist_ok=True)

        for name, model in self.base_models.items():
            joblib.dump(model, base_path / f"{name}_model.pkl")

        # Guardar modelos ensemble
        ensemble_path = save_path / "ensemble_models"
        ensemble_path.mkdir(exist_ok=True)

        for name, model in self.ensemble_models.items():
            joblib.dump(model, ensemble_path / f"{name}_ensemble.pkl")

        # Guardar pesos
        if self.model_weights:
            import json

            with open(save_path / "model_weights.json", "w") as f:
                json.dump(self.model_weights, f, indent=2)

        print(f"💾 Modelos guardados en: {save_path}")


class WeightedEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble ponderado personalizado"""

    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        """
        Inicializa el ensemble ponderado

        Args:
            models: Diccionario con modelos
            weights: Diccionario con pesos para cada modelo
        """
        self.models = models
        self.weights = weights
        self.classes_ = None

    def fit(self, X, y):
        """Ajusta el ensemble (los modelos ya están entrenados)"""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        """Hace predicciones usando votación ponderada"""
        X = check_array(X)
        predictions = []

        for sample in X:
            sample = sample.reshape(1, -1)
            weighted_votes = defaultdict(float)

            for name, model in self.models.items():
                if hasattr(model, "predict"):
                    pred = model.predict(sample)[0]
                    weight = self.weights.get(name, 0)
                    weighted_votes[pred] += weight

            best_prediction = max(
                weighted_votes.keys(), key=lambda x: weighted_votes[x]
            )
            predictions.append(best_prediction)

        return np.array(predictions)

    def predict_proba(self, X):
        """Calcula probabilidades usando promedio ponderado"""
        X = check_array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))

        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                weight = self.weights.get(name, 0) / total_weight
                model_proba = model.predict_proba(X)
                probabilities += weight * model_proba

        return probabilities

    def score(self, X, y):
        """Calcula la precisión del ensemble"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class DynamicEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble dinámico que selecciona modelos por confianza"""

    def __init__(self, models: Dict[str, Any], confidence_threshold: float = 0.7):
        """
        Inicializa el ensemble dinámico

        Args:
            models: Diccionario con modelos
            confidence_threshold: Umbral de confianza
        """
        self.models = models
        self.confidence_threshold = confidence_threshold
        self.classes_ = None

    def fit(self, X, y):
        """Ajusta el ensemble (los modelos ya están entrenados)"""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        """Hace predicciones seleccionando dinámicamente los modelos"""
        X = check_array(X)
        predictions = []

        for sample in X:
            sample = sample.reshape(1, -1)
            confident_predictions = []

            # Buscar modelos con alta confianza
            for name, model in self.models.items():
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(sample)[0]
                    max_proba = np.max(proba)

                    if max_proba >= self.confidence_threshold:
                        pred = model.predict(sample)[0]
                        confident_predictions.append((pred, max_proba))

            if confident_predictions:
                # Usar la predicción con mayor confianza
                best_pred = max(confident_predictions, key=lambda x: x[1])[0]
                predictions.append(best_pred)
            else:
                # Si ningún modelo es confiable, usar votación simple
                votes = []
                for name, model in self.models.items():
                    if hasattr(model, "predict"):
                        pred = model.predict(sample)[0]
                        votes.append(pred)

                if votes:
                    # Votación por mayoría
                    from collections import Counter

                    vote_counts = Counter(votes)
                    best_pred = vote_counts.most_common(1)[0][0]
                    predictions.append(best_pred)
                else:
                    predictions.append(0)  # Fallback

        return np.array(predictions)

    def score(self, X, y):
        """Calcula la precisión del ensemble"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class TemporalEnsemble:
    """Ensemble temporal para secuencias de poses"""

    def __init__(self, models: Dict[str, Any], window_size: int = 5):
        """
        Inicializa el ensemble temporal

        Args:
            models: Diccionario con modelos
            window_size: Tamaño de ventana temporal
        """
        self.models = models
        self.window_size = window_size
        self.prediction_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)

    def predict_temporal(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Hace predicción temporal considerando historial

        Args:
            X: Características de la muestra actual

        Returns:
            Tupla (predicción, confianza)
        """
        current_predictions = {}
        current_confidences = {}

        # Obtener predicciones actuales de todos los modelos
        for name, model in self.models.items():
            if hasattr(model, "predict") and hasattr(model, "predict_proba"):
                pred = model.predict(X.reshape(1, -1))[0]
                proba = model.predict_proba(X.reshape(1, -1))[0]
                confidence = np.max(proba)

                current_predictions[name] = pred
                current_confidences[name] = confidence

        # Añadir al historial
        self.prediction_history.append(current_predictions)
        self.confidence_history.append(current_confidences)

        # Realizar predicción temporal
        if len(self.prediction_history) < self.window_size:
            # No hay suficiente historial, usar votación simple
            votes = list(current_predictions.values())
            from collections import Counter

            vote_counts = Counter(votes)
            final_prediction = vote_counts.most_common(1)[0][0]
            avg_confidence = np.mean(list(current_confidences.values()))
        else:
            # Usar ventana temporal completa
            final_prediction, avg_confidence = self._temporal_smoothing()

        return final_prediction, avg_confidence

    def _temporal_smoothing(self) -> Tuple[int, float]:
        """Suavizado temporal de predicciones"""
        # Contar votos ponderados por confianza y recencia
        weighted_votes = defaultdict(float)
        total_weight = 0

        for i, (predictions, confidences) in enumerate(
            zip(self.prediction_history, self.confidence_history)
        ):
            # Peso por recencia (predicciones más recientes tienen más peso)
            time_weight = (i + 1) / len(self.prediction_history)

            for model_name in predictions:
                pred = predictions[model_name]
                conf = confidences[model_name]

                # Peso combinado: confianza * recencia
                weight = conf * time_weight
                weighted_votes[pred] += weight
                total_weight += weight

        # Normalizar y seleccionar mejor predicción
        if total_weight > 0:
            for pred in weighted_votes:
                weighted_votes[pred] /= total_weight

        final_prediction = max(weighted_votes.keys(), key=lambda x: weighted_votes[x])
        avg_confidence = weighted_votes[final_prediction]

        return final_prediction, avg_confidence

    def reset_history(self):
        """Reinicia el historial temporal"""
        self.prediction_history.clear()
        self.confidence_history.clear()
