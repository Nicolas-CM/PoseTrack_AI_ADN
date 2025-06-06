"""
Data analysis and visualization utilities

This module provides functions for analyzing model performance, visualizing results,
and interpreting machine learning models for activity recognition. It includes
tools for plotting confusion matrices, feature importance, and dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], title: str = "Matriz de Confusión",
                         save_path: Optional[str] = None):
    """
    Visualize a confusion matrix
    
    This function creates a heatmap visualization of the confusion matrix to 
    evaluate model performance. It shows how well the model classifies each activity
    and where misclassifications occur.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names for axis labels
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance(model, feature_names: List[str], 
                          top_n: int = 20, title: str = "Importancia de Características",
                          save_path: Optional[str] = None):
    """
    Visualiza la importancia de características de un modelo
    
    Args:
        model: Modelo entrenado con feature_importances_
        feature_names: Nombres de las características
        top_n: Número de características principales a mostrar
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
    """
    if not hasattr(model, 'feature_importances_'):
        print("El modelo no tiene información de importancia de características")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    
    plt.title(title)
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Características')
    plt.ylabel('Importancia')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_activity_distribution(labels: List[str], title: str = "Distribución de Actividades",
                             save_path: Optional[str] = None):
    """
    Visualiza la distribución de actividades en el dataset
    
    Args:
        labels: Lista de etiquetas de actividades
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
    """
    plt.figure(figsize=(10, 6))
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    plt.bar(unique_labels, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title(title)
    plt.xlabel('Actividades')
    plt.ylabel('Número de Muestras')
    plt.xticks(rotation=45, ha='right')
    
    # Añadir valores encima de las barras
    for i, count in enumerate(counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_correlation(features: np.ndarray, feature_names: List[str],
                           title: str = "Correlación entre Características",
                           save_path: Optional[str] = None):
    """
    Visualiza la correlación entre características
    
    Args:
        features: Matriz de características
        feature_names: Nombres de las características
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
    """
    # Tomar muestra si hay demasiadas características
    if len(feature_names) > 50:
        print(f"Demasiadas características ({len(feature_names)}), mostrando solo las primeras 50")
        features = features[:, :50]
        feature_names = feature_names[:50]
    
    df = pd.DataFrame(features, columns=feature_names)
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(15, 12))
    
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.1)
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_pca_analysis(features: np.ndarray, labels: np.ndarray,
                     title: str = "Análisis PCA", save_path: Optional[str] = None):
    """
    Visualiza análisis de componentes principales
    
    Args:
        features: Matriz de características
        labels: Etiquetas de clase
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
    """
    # Aplicar PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    
    # Obtener clases únicas
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
    plt.title(f'{title}\nVarianza explicada total: {sum(pca.explained_variance_ratio_):.2%}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return pca

def plot_tsne_analysis(features: np.ndarray, labels: np.ndarray,
                      title: str = "Análisis t-SNE", save_path: Optional[str] = None):
    """
    Visualiza análisis t-SNE
    
    Args:
        features: Matriz de características
        labels: Etiquetas de clase
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
    """
    # Aplicar t-SNE (usar muestra si dataset es muy grande)
    if len(features) > 1000:
        print("Dataset grande, usando muestra para t-SNE")
        indices = np.random.choice(len(features), 1000, replace=False)
        features_sample = features[indices]
        labels_sample = labels[indices]
    else:
        features_sample = features
        labels_sample = labels
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features_sample)
    
    plt.figure(figsize=(12, 8))
    
    # Obtener clases únicas
    unique_labels = np.unique(labels_sample)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_sample == label
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=50)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_learning_curves(train_scores: List[float], val_scores: List[float],
                        epochs: List[int], title: str = "Curvas de Aprendizaje",
                        save_path: Optional[str] = None):
    """
    Visualiza curvas de aprendizaje
    
    Args:
        train_scores: Puntuaciones de entrenamiento
        val_scores: Puntuaciones de validación
        epochs: Épocas o iteraciones
        title: Título del gráfico
        save_path: Ruta para guardar el gráfico (opcional)
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_scores, 'o-', label='Entrenamiento', color='blue')
    plt.plot(epochs, val_scores, 'o-', label='Validación', color='red')
    
    plt.xlabel('Época')
    plt.ylabel('Puntuación')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_model_report(model_path: str, X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: List[str], class_names: List[str],
                         save_dir: str = None):
    """
    Genera un reporte completo de evaluación de modelo
    
    Args:
        model_path: Ruta al modelo guardado
        X_test: Características de prueba
        y_test: Etiquetas de prueba
        feature_names: Nombres de características
        class_names: Nombres de clases
        save_dir: Directorio para guardar gráficos (opcional)
    """
    # Cargar modelo
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    label_encoder = model_data.get('label_encoder')
    
    print(f"=== REPORTE DE EVALUACIÓN ===")
    print(f"Modelo: {model_path}")
    print(f"Tipo: {type(model).__name__}")
    
    # Normalizar características si es necesario
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    # Codificar etiquetas si es necesario
    if label_encoder:
        y_test_encoded = label_encoder.transform(y_test)
        class_names_encoded = label_encoder.classes_
    else:
        y_test_encoded = y_test
        class_names_encoded = class_names
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    
    # Reporte de clasificación
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names_encoded))
    
    # Visualizaciones
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Matriz de confusión
        plot_confusion_matrix(y_test_encoded, y_pred, class_names_encoded,
                            save_path=save_dir / "confusion_matrix.png")
        
        # Importancia de características (si está disponible)
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names,
                                  save_path=save_dir / "feature_importance.png")
        
        # Análisis PCA
        plot_pca_analysis(X_test_scaled, y_test_encoded,
                        save_path=save_dir / "pca_analysis.png")
    
    else:
        # Mostrar sin guardar
        plot_confusion_matrix(y_test_encoded, y_pred, class_names_encoded)
        
        if hasattr(model, 'feature_importances_'):
            plot_feature_importance(model, feature_names)
        
        plot_pca_analysis(X_test_scaled, y_test_encoded)

def analyze_dataset_statistics(features: np.ndarray, labels: np.ndarray,
                             feature_names: List[str]):
    """
    Analiza estadísticas básicas del dataset
    
    Args:
        features: Matriz de características
        labels: Etiquetas
        feature_names: Nombres de características
    """
    print("=== ESTADÍSTICAS DEL DATASET ===")
    print(f"Número de muestras: {len(features)}")
    print(f"Número de características: {features.shape[1]}")
    print(f"Número de clases: {len(np.unique(labels))}")
    
    print("\n--- Distribución de Clases ---")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(labels)) * 100
        print(f"{label}: {count} muestras ({percentage:.1f}%)")
    
    print("\n--- Estadísticas de Características ---")
    print(f"Media general: {np.mean(features):.4f}")
    print(f"Desviación estándar: {np.std(features):.4f}")
    print(f"Valor mínimo: {np.min(features):.4f}")
    print(f"Valor máximo: {np.max(features):.4f}")
    
    # Características con mayor varianza
    variances = np.var(features, axis=0)
    top_variance_indices = np.argsort(variances)[-10:][::-1]
    
    print("\n--- Top 10 Características con Mayor Varianza ---")
    for i, idx in enumerate(top_variance_indices):
        if idx < len(feature_names):
            print(f"{i+1}. {feature_names[idx]}: {variances[idx]:.6f}")
    
    # Detectar posibles características problemáticas
    zero_variance = np.sum(variances < 1e-8)
    if zero_variance > 0:
        print(f"\n⚠️  Encontradas {zero_variance} características con varianza casi cero")
    
    # Detectar valores NaN o infinitos
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    
    if nan_count > 0:
        print(f"⚠️  Encontrados {nan_count} valores NaN")
    
    if inf_count > 0:
        print(f"⚠️  Encontrados {inf_count} valores infinitos")
