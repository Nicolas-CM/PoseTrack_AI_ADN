# 🎯 PoseTrack AI - Sistema de Análisis de Movimiento en Tiempo Real

## Final Project – Artificial Intelligence 1

**Universidad Icesi – Department of Intelligent Computing and Systems**  
**Semester 2025-1**

## 📖 Descripción

PoseTrack AI es una aplicación completa de análisis de video en tiempo real que detecta y clasifica actividades humanas simples utilizando **MediaPipe** para el seguimiento postural y **Machine Learning** para la clasificación de actividades. El sistema puede detectar las siguientes actividades:

- **Acercarse** hacia la cámara
- **Alejarse** de la cámara  
- **Girar a la derecha** (girarD)
- **Girar a la izquierda** (girarI)
- **Sentarse**
- **Levantarse**

## ✨ Características Principales

### 🎥 Análisis en Tiempo Real
- Seguimiento de pose con 15 landmarks clave del cuerpo
- Extracción automática de características posturales y de movimiento
- Clasificación de actividades en tiempo real con modelos ML entrenados

### 🤖 Machine Learning Avanzado
- **3 tipos de modelos**: SVM, Random Forest, XGBoost
- **Sistema de características sofisticado**: estadísticas, velocidades, ángulos articulares, trayectorias, análisis de frecuencia
- **Ventana deslizante** de 30 frames para análisis temporal
- **Suavizado de predicciones** para mayor estabilidad

### 🖥️ Interfaz Gráfica Completa
- **Video en tiempo real** con overlay de landmarks
- **Métricas posturales** en tiempo real
- **Selección dinámica** de modelos ML
- **Sistema de entrenamiento** integrado
- **Visualización de confianza** de predicciones

## 📊 Resultados de Entrenamiento

El sistema ha sido entrenado con **51 videos** (6,393 muestras) y ha alcanzado los siguientes resultados:

| Modelo | Precisión | Validación Cruzada | Desviación Estándar |
|--------|-----------|-------------------|---------------------|
| **Random Forest** | **100.0%** | **100.0%** | **0.000** |
| **XGBoost** | **100.0%** | **99.9%** | **0.001** |
| **SVM** | **90.8%** | **91.7%** | **0.004** |

### 📈 Distribución del Dataset
- **Sentarse**: 1,579 muestras (24.7%)
- **Levantarse**: 1,240 muestras (19.4%)
- **Girar Derecha**: 1,149 muestras (18.0%)
- **Acercarse**: 919 muestras (14.4%)
- **Alejarse**: 892 muestras (14.0%)
- **Girar Izquierda**: 614 muestras (9.6%)

## 🚀 Instalación y Uso

### 1. Verificación del Sistema
```bash
python setup.py
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Entrenar Modelos (opcional)
```bash
python main.py train
```

### 4. Ejecutar la Aplicación
```bash
python main.py
```

## 📁 Estructura del Proyecto

```
PoseTrack_AI_ADN/
├── main.py                          # Punto de entrada principal
├── setup.py                         # Script de verificación del sistema
├── requirements.txt                 # Dependencias del proyecto
├── README.md                        # Documentación
├── config/
│   └── settings.py                  # Configuración centralizada
├── src/
│   ├── core/
│   │   ├── pose_tracker.py          # Seguimiento de pose con MediaPipe
│   │   ├── feature_extractor.py     # Extracción de características
│   │   └── activity_classifier.py   # Clasificación de actividades
│   ├── gui/
│   │   └── main_gui.py              # Interfaz gráfica principal
│   ├── training/
│   │   └── train_model.py           # Sistema de entrenamiento
│   └── utils/
│       ├── video_utils.py           # Utilidades de video
│       └── analysis_utils.py        # Utilidades de análisis
├── models/                          # Modelos entrenados
├── data/                            # Datos de entrenamiento
└── Entrega 2/Videos/               # Videos de entrenamiento (51 videos)
```

## 🔧 Requisitos del Sistema

### Software
- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows, macOS, Linux
- **Cámara**: Webcam o cámara USB

### Hardware Recomendado
- **CPU**: Multi-core moderno
- **RAM**: 8GB mínimo, 16GB recomendado
- **Cámara**: 720p o superior para mejor detección

## 🎮 Uso de la Interfaz

### Panel Principal
1. **Video en Tiempo Real**: Muestra la cámara con landmarks superpuestos
2. **Actividad Detectada**: Muestra la actividad actual y confianza
3. **Métricas Posturales**: Información en tiempo real sobre la postura

### Controles
- **Seleccionar Modelo**: Cambiar entre SVM, Random Forest, XGBoost
- **Entrenar Nuevo Modelo**: Abrir diálogo de entrenamiento
- **Configuración**: Ajustar parámetros del sistema
- **Información**: Ver detalles del modelo actual

## 🧠 Características Técnicas

### Extracción de Características (281 características totales)
- **Estadísticas básicas**: Media, desviación estándar, min, max
- **Velocidades**: Velocidad de landmarks individuales y centro de masa
- **Ángulos articulares**: 7 ángulos clave del cuerpo
- **Trayectorias**: Análisis de movimiento direccional
- **Análisis de frecuencia**: Componentes de frecuencia dominantes

### Landmarks Utilizados (15 puntos clave)
- Nariz, ojos, orejas
- Hombros, codos, muñecas
- Caderas, rodillas, tobillos

### Algoritmos de ML
- **SVM**: Support Vector Machine con kernel RBF
- **Random Forest**: 100 árboles con optimización automática
- **XGBoost**: Gradient boosting con regularización

## 📈 Rendimiento del Sistema

### Tiempos de Procesamiento
- **Detección de pose**: ~15-20ms por frame
- **Extracción de características**: ~2-3ms por frame
- **Clasificación**: <1ms por predicción
- **FPS típico**: 25-30 FPS en hardware moderno

### Uso de Recursos
- **CPU**: 15-25% en procesamiento normal
- **RAM**: ~200-300MB durante ejecución
- **GPU**: Opcional, acelera MediaPipe si está disponible

## 🔄 Actualizaciones y Extensiones

### Agregar Nuevas Actividades
1. Grabar videos de la nueva actividad
2. Colocar en directorio apropiado
3. Ejecutar `python main.py train`
4. El sistema detectará automáticamente la nueva clase

## 📚 Referencias Técnicas

### Bibliotecas Principales
- **[MediaPipe](https://mediapipe.dev/)**: Framework de ML para percepción multimedia
- **[OpenCV](https://opencv.org/)**: Biblioteca de visión por computadora
- **[scikit-learn](https://scikit-learn.org/)**: Herramientas de ML en Python
- **[XGBoost](https://xgboost.readthedocs.io/)**: Gradient boosting optimizado

### Algoritmos y Técnicas
- **Pose Estimation**: BlazePose (MediaPipe)
- **Feature Engineering**: Análisis temporal con ventana deslizante
- **Classification**: Ensemble methods y SVM
- **Signal Processing**: Filtros de suavizado temporal

## 👥 Autores

- [Davide Flamini](https://github.com/davidone007)
- [Andrés Cabezas](https://github.com/andrescabezas26)
- [Nicolas Cuellar](https://github.com/Nicolas-CM)

### Useful Links

* 🔗 **Project Repository:** [PoseTrack_AI_ADN](https://github.com/Nicolas-CM/PoseTrack_AI_ADN.git)  
* 📄 **MMFit Dataset:** [Official Site](https://mmfit.github.io) | [GitHub Repository](https://github.com/KDMStromback/mm-fit)

## 📄 Licencia

Este proyecto es desarrollado con fines académicos en la Universidad Icesi.

---

**Desarrollado con ❤️ para el análisis de movimiento humano en tiempo real**