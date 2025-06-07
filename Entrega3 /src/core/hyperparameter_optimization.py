"""
Módulo de optimización de hiperparámetros avanzada
Implementa búsqueda bayesiana, Optuna y optimización automática
"""

import numpy as np
import optuna
from typing import Dict, Any, List, Tuple, Optional, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import joblib
from pathlib import Path
import json
import time

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("scikit-optimize no disponible. Solo se usará Optuna.")


class HyperparameterOptimizer:
    """Optimizador de hiperparámetros usando múltiples algoritmos"""

    def __init__(
        self,
        n_trials: int = 20,
        cv_folds: int = 3,
        scoring: str = "accuracy",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Inicializa el optimizador

        Args:
            n_trials: Número de pruebas para optimización
            cv_folds: Número de folds para validación cruzada
            scoring: Métrica de evaluación
            random_state: Semilla aleatoria
            n_jobs: Número de trabajos paralelos
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params = {}
        self.optimization_history = {}

        # Configurar Optuna para suprimir logs verbosos
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optimize_random_forest_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimiza Random Forest usando Optuna

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento

        Returns:
            Diccionario con mejores parámetros y puntuación
        """

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": self.random_state,
            }

            if not params["bootstrap"]:
                params["oob_score"] = False

            model = RandomForestClassifier(**params)

            # Validación cruzada
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                ),
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )

            return cv_scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Callback para mostrar progreso
        def progress_callback(study, trial):
            print(
                f"  📊 Trial {trial.number + 1}/{self.n_trials}: Score = {trial.value:.4f}"
            )

        study.optimize(objective, n_trials=self.n_trials, callbacks=[progress_callback])

        self.best_params["random_forest"] = study.best_params
        self.optimization_history["random_forest"] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }

    def optimize_xgboost_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimiza XGBoost usando Optuna

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento

        Returns:
            Diccionario con mejores parámetros y puntuación
        """

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 1.0, log=True),
                "random_state": self.random_state,
                "verbosity": 0,
            }

            model = xgb.XGBClassifier(**params)

            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                ),
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )

            return cv_scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Callback para mostrar progreso
        def progress_callback(study, trial):
            print(
                f"  📊 Trial {trial.number + 1}/{self.n_trials}: Score = {trial.value:.4f}"
            )

        study.optimize(objective, n_trials=self.n_trials, callbacks=[progress_callback])

        self.best_params["xgboost"] = study.best_params
        self.optimization_history["xgboost"] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }

    def optimize_svm_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimiza SVM usando Optuna

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento

        Returns:
            Diccionario con mejores parámetros y puntuación
        """

        def objective(trial):
            kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])

            params = {
                "kernel": kernel,
                "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                "random_state": self.random_state,
                "probability": True,
            }

            if kernel == "rbf" or kernel == "sigmoid":
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            elif kernel == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])

            model = SVC(**params)

            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                ),
                scoring=self.scoring,
                n_jobs=1,  # SVM no es thread-safe en algunos casos
            )

            return cv_scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Callback para mostrar progreso
        def progress_callback(study, trial):
            print(
                f"  📊 Trial {trial.number + 1}/{self.n_trials}: Score = {trial.value:.4f}"
            )

        study.optimize(objective, n_trials=self.n_trials, callbacks=[progress_callback])

        self.best_params["svm"] = study.best_params
        self.optimization_history["svm"] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }

    def optimize_mlp_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimiza MLP (Red Neuronal) usando Optuna

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento

        Returns:
            Diccionario con mejores parámetros y puntuación
        """

        def objective(trial):
            # Arquitectura de la red
            n_layers = trial.suggest_int("n_layers", 1, 4)
            hidden_layer_sizes = []

            for i in range(n_layers):
                layer_size = trial.suggest_int(f"layer_{i}_size", 32, 512)
                hidden_layer_sizes.append(layer_size)

            params = {
                "hidden_layer_sizes": tuple(hidden_layer_sizes),
                "activation": trial.suggest_categorical(
                    "activation", ["relu", "tanh", "logistic"]
                ),
                "solver": trial.suggest_categorical("solver", ["adam", "lbfgs"]),
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
                "learning_rate": trial.suggest_categorical(
                    "learning_rate", ["constant", "invscaling", "adaptive"]
                ),
                "max_iter": 1000,
                "random_state": self.random_state,
                "early_stopping": True,
                "validation_fraction": 0.1,
            }

            if params["solver"] == "adam":
                params["learning_rate_init"] = trial.suggest_float(
                    "learning_rate_init", 1e-5, 1e-1, log=True
                )

            model = MLPClassifier(**params)

            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                ),
                scoring=self.scoring,
                n_jobs=1,  # MLP puede tener problemas con paralelización
            )

            return cv_scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Callback para mostrar progreso
        def progress_callback(study, trial):
            print(
                f"  📊 Trial {trial.number + 1}/{min(self.n_trials, 50)}: Score = {trial.value:.4f}"
            )

        study.optimize(
            objective, n_trials=min(self.n_trials, 50), callbacks=[progress_callback]
        )  # Menos trials para MLP

        self.best_params["mlp"] = study.best_params
        self.optimization_history["mlp"] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }

    def optimize_lightgbm_optuna(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimiza LightGBM usando Optuna

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento

        Returns:
            Diccionario con mejores parámetros y puntuación
        """

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 1.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "random_state": self.random_state,
                "verbosity": -1,
            }

            model = lgb.LGBMClassifier(**params)

            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                ),
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )

            return cv_scores.mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        # Callback para mostrar progreso
        def progress_callback(study, trial):
            print(
                f"  📊 Trial {trial.number + 1}/{self.n_trials}: Score = {trial.value:.4f}"
            )

        study.optimize(objective, n_trials=self.n_trials, callbacks=[progress_callback])

        self.best_params["lightgbm"] = study.best_params
        self.optimization_history["lightgbm"] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
        }

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }

    def optimize_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        models_to_optimize: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimiza todos los modelos especificados

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            models_to_optimize: Lista de modelos a optimizar

        Returns:
            Diccionario con resultados de optimización para cada modelo
        """
        if models_to_optimize is None:
            models_to_optimize = ["random_forest", "xgboost", "svm", "mlp", "lightgbm"]

        optimization_results = {}

        print("🚀 Iniciando optimización de hiperparámetros...")
        print(f"Modelos a optimizar: {models_to_optimize}")
        print(f"Número de trials por modelo: {self.n_trials}")
        print(f"Validación cruzada: {self.cv_folds} folds")
        print("=" * 60)

        for model_name in models_to_optimize:
            print(f"\n🔧 Optimizando {model_name.upper()}...")
            start_time = time.time()

            try:
                if model_name == "random_forest":
                    result = self.optimize_random_forest_optuna(X_train, y_train)
                elif model_name == "xgboost":
                    result = self.optimize_xgboost_optuna(X_train, y_train)
                elif model_name == "svm":
                    result = self.optimize_svm_optuna(X_train, y_train)
                elif model_name == "mlp":
                    result = self.optimize_mlp_optuna(X_train, y_train)
                elif model_name == "lightgbm":
                    result = self.optimize_lightgbm_optuna(X_train, y_train)
                else:
                    print(f"❌ Modelo no soportado: {model_name}")
                    continue

                optimization_results[model_name] = result
                elapsed_time = time.time() - start_time

                print(f"✅ {model_name.upper()} optimizado")
                print(f"   Mejor puntuación: {result['best_score']:.4f}")
                print(f"   Tiempo: {elapsed_time:.2f}s")

            except Exception as e:
                print(f"❌ Error optimizando {model_name}: {str(e)}")
                continue

        return optimization_results

    def create_optimized_models(
        self, optimization_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Crea modelos con los parámetros optimizados

        Args:
            optimization_results: Resultados de optimización

        Returns:
            Diccionario con modelos optimizados
        """
        optimized_models = {}

        for model_name, result in optimization_results.items():
            params = result["best_params"].copy()

            if model_name == "random_forest":
                params["random_state"] = self.random_state
                optimized_models[model_name] = RandomForestClassifier(**params)

            elif model_name == "xgboost":
                params["random_state"] = self.random_state
                params["verbosity"] = 0
                optimized_models[model_name] = xgb.XGBClassifier(**params)

            elif model_name == "svm":
                params["random_state"] = self.random_state
                params["probability"] = True
                optimized_models[model_name] = SVC(**params)

            elif model_name == "mlp":
                params["random_state"] = self.random_state
                params["max_iter"] = 1000
                # Filtrar parámetros que no son válidos para MLPClassifier
                mlp_params = {
                    k: v
                    for k, v in params.items()
                    if not k.startswith("n_layers") and not k.startswith("layer_")
                }
                optimized_models[model_name] = MLPClassifier(**mlp_params)

            elif model_name == "lightgbm":
                params["random_state"] = self.random_state
                params["verbosity"] = -1
                optimized_models[model_name] = lgb.LGBMClassifier(**params)

        return optimized_models

    def save_optimization_results(self, save_path: str):
        """
        Guarda los resultados de optimización

        Args:
            save_path: Ruta donde guardar los resultados
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Guardar parámetros óptimos
        with open(save_path / "best_params.json", "w") as f:
            json.dump(self.best_params, f, indent=2)

        # Guardar historial de optimización
        with open(save_path / "optimization_history.json", "w") as f:
            json.dump(self.optimization_history, f, indent=2)

        print(f"💾 Resultados de optimización guardados en: {save_path}")

    def load_optimization_results(self, load_path: str) -> bool:
        """
        Carga resultados de optimización previamente guardados

        Args:
            load_path: Ruta de donde cargar los resultados

        Returns:
            True si se cargaron exitosamente, False en caso contrario
        """
        load_path = Path(load_path)

        try:
            # Cargar parámetros óptimos
            with open(load_path / "best_params.json", "r") as f:
                self.best_params = json.load(f)

            # Cargar historial de optimización
            with open(load_path / "optimization_history.json", "r") as f:
                self.optimization_history = json.load(f)

            print(f"📁 Resultados de optimización cargados desde: {load_path}")
            return True

        except Exception as e:
            print(f"❌ Error cargando resultados de optimización: {str(e)}")
            return False


class AutoMLPipeline:
    """Pipeline automatizado de machine learning"""

    def __init__(
        self,
        optimizer: HyperparameterOptimizer,
        models_to_try: Optional[List[str]] = None,
        ensemble_methods: Optional[List[str]] = None,
    ):
        """
        Inicializa el pipeline AutoML

        Args:
            optimizer: Optimizador de hiperparámetros
            models_to_try: Lista de modelos a probar
            ensemble_methods: Métodos de ensemble a usar
        """
        self.optimizer = optimizer
        self.models_to_try = models_to_try or [
            "random_forest",
            "xgboost",
            "lightgbm",
            "svm",
        ]
        self.ensemble_methods = ensemble_methods or ["voting", "stacking"]
        self.trained_models = {}
        self.ensemble_models = {}
        self.performance_results = {}

    def auto_train_and_optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Entrena y optimiza automáticamente múltiples modelos

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación
            y_val: Etiquetas de validación

        Returns:
            Diccionario con resultados del AutoML
        """
        print("🤖 Iniciando pipeline AutoML...")

        # 1. Optimización de hiperparámetros
        optimization_results = self.optimizer.optimize_all_models(
            X_train, y_train, self.models_to_try
        )

        # 2. Crear modelos optimizados
        optimized_models = self.optimizer.create_optimized_models(optimization_results)

        # 3. Entrenar modelos optimizados
        print("\n🏋️ Entrenando modelos optimizados...")
        for model_name, model in optimized_models.items():
            print(f"   Entrenando {model_name}...")
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model

            # Evaluar modelo
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)

            self.performance_results[model_name] = {
                "train_score": train_score,
                "val_score": val_score,
                "optimization_score": optimization_results[model_name]["best_score"],
            }

        # 4. Crear modelos ensemble
        if len(self.trained_models) > 1:
            print("\n🎭 Creando modelos ensemble...")
            self._create_ensemble_models(X_train, y_train, X_val, y_val)

        # 5. Seleccionar mejor modelo
        best_model_name = self._select_best_model()

        results = {
            "trained_models": self.trained_models,
            "ensemble_models": self.ensemble_models,
            "performance_results": self.performance_results,
            "optimization_results": optimization_results,
            "best_model_name": best_model_name,
            "best_model": self.trained_models.get(best_model_name)
            or self.ensemble_models.get(best_model_name),
        }

        print(f"\n🏆 Mejor modelo: {best_model_name}")
        if best_model_name in self.performance_results:
            print(
                f"   Puntuación de validación: {self.performance_results[best_model_name]['val_score']:.4f}"
            )

        return results

    def _create_ensemble_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Crea modelos ensemble"""
        from sklearn.ensemble import VotingClassifier, StackingClassifier

        base_models = [(name, model) for name, model in self.trained_models.items()]

        if "voting" in self.ensemble_methods:
            # Ensemble de votación
            voting_clf = VotingClassifier(estimators=base_models, voting="soft")
            voting_clf.fit(X_train, y_train)
            self.ensemble_models["voting"] = voting_clf

            # Evaluar ensemble de votación
            train_score = voting_clf.score(X_train, y_train)
            val_score = voting_clf.score(X_val, y_val)

            self.performance_results["voting"] = {
                "train_score": train_score,
                "val_score": val_score,
                "optimization_score": val_score,
            }

        if "stacking" in self.ensemble_methods:
            # Ensemble de stacking
            stacking_clf = StackingClassifier(
                estimators=base_models,
                final_estimator=RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
                cv=3,
            )
            stacking_clf.fit(X_train, y_train)
            self.ensemble_models["stacking"] = stacking_clf

            # Evaluar ensemble de stacking
            train_score = stacking_clf.score(X_train, y_train)
            val_score = stacking_clf.score(X_val, y_val)

            self.performance_results["stacking"] = {
                "train_score": train_score,
                "val_score": val_score,
                "optimization_score": val_score,
            }

    def _select_best_model(self) -> str:
        """Selecciona el mejor modelo basado en puntuación de validación"""
        best_score = -1
        best_model_name = None

        for model_name, results in self.performance_results.items():
            val_score = results["val_score"]
            if val_score > best_score:
                best_score = val_score
                best_model_name = model_name

        return best_model_name or "random_forest"
