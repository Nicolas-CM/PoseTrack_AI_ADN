#!/usr/bin/env python3
"""
Script de configuración y verificación del sistema PoseTrack AI
"""

import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Verifica la versión de Python"""
    print("🐍 Verificando versión de Python...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} no es compatible")
        print("   Se requiere Python 3.8 o superior")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} es compatible")
    return True


def install_dependencies():
    """Instala las dependencias del proyecto"""
    print("\n📦 Instalando dependencias...")

    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Archivo requirements.txt no encontrado")
        return False

    try:
        # Instalar dependencias
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Dependencias instaladas exitosamente")
            return True
        else:
            print(f"❌ Error instalando dependencias: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error ejecutando pip: {e}")
        return False


def check_camera():
    """Verifica que la cámara esté disponible"""
    print("\n📹 Verificando acceso a cámara...")

    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                print("✅ Cámara detectada y funcionando")
                return True
            else:
                print("⚠️  Cámara detectada pero no puede capturar frames")
                return False
        else:
            print("❌ No se pudo acceder a la cámara")
            print("   Verifica que no esté siendo usada por otra aplicación")
            return False

    except ImportError:
        print("❌ OpenCV no está instalado")
        return False
    except Exception as e:
        print(f"❌ Error verificando cámara: {e}")
        return False


def check_mediapipe():
    """Verifica que MediaPipe esté funcionando"""
    print("\n🤖 Verificando MediaPipe...")

    try:
        import mediapipe as mp

        # Intentar inicializar pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        pose.close()

        print("✅ MediaPipe funciona correctamente")
        return True

    except ImportError:
        print("❌ MediaPipe no está instalado")
        return False
    except Exception as e:
        print(f"❌ Error con MediaPipe: {e}")
        return False


def check_videos():
    """Verifica que los videos de entrenamiento estén disponibles"""
    print("\n🎥 Verificando videos de entrenamiento...")

    videos_dir = Path("/Videos")
    if not videos_dir.exists():
        print(f"❌ Directorio de videos no encontrado: {videos_dir}")
        return False

    # Contar videos por actividad
    activities = ["acercarse", "alejarse", "girarD", "girarI", "sentarse", "levantarse"]
    video_count = {}

    for activity in activities:
        count = len(list(videos_dir.glob(f"{activity}*")))
        video_count[activity] = count

    total_videos = sum(video_count.values())

    if total_videos == 0:
        print("❌ No se encontraron videos de entrenamiento")
        return False

    print(f"✅ Encontrados {total_videos} videos de entrenamiento:")
    for activity, count in video_count.items():
        print(f"   {activity}: {count} videos")

    return True


def check_directories():
    """Verifica y crea directorios necesarios"""
    print("\n📁 Verificando estructura de directorios...")

    required_dirs = [
        "models",
        "data",
        "config",
        "src/core",
        "src/gui",
        "src/training",
        "src/utils",
    ]

    all_exist = True

    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"📁 Creando directorio: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ Directorio existe: {dir_path}")

    return True


def check_gpu():
    """Verifica disponibilidad de GPU para aceleración (opcional)"""
    print("\n🚀 Verificando aceleración GPU (opcional)...")

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            print(f"✅ Encontradas {len(gpus)} GPU(s) disponibles")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("ℹ️  No se encontraron GPUs (se usará CPU)")

    except ImportError:
        print("ℹ️  TensorFlow no instalado (GPU no disponible)")
    except Exception as e:
        print(f"⚠️  Error verificando GPU: {e}")


def run_quick_test():
    """Ejecuta una prueba rápida del sistema"""
    print("\n🧪 Ejecutando prueba rápida del sistema...")

    try:
        # Importar módulos principales
        sys.path.append(str(Path.cwd()))

        from src.core.pose_tracker import PoseTracker
        from src.core.feature_extractor import FeatureExtractor
        from src.core.activity_classifier import ActivityClassifier

        # Probar pose tracker
        pose_tracker = PoseTracker()
        print("✅ PoseTracker inicializado")

        # Probar feature extractor
        feature_extractor = FeatureExtractor()
        print("✅ FeatureExtractor inicializado")

        # Probar classifier
        classifier = ActivityClassifier()
        print("✅ ActivityClassifier inicializado")

        # Limpiar
        pose_tracker.close()

        print("✅ Prueba rápida completada exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error en prueba rápida: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Función principal de configuración"""
    print("🎯 PoseTrack AI - Configuración del Sistema")
    print("=" * 50)

    # Lista de verificaciones
    checks = [
        ("Python", check_python_version),
        ("Directorios", check_directories),
        ("Dependencias", install_dependencies),
        ("MediaPipe", check_mediapipe),
        ("Cámara", check_camera),
        ("Videos", check_videos),
        ("Sistema", run_quick_test),
    ]

    results = {}

    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ Error en verificación {name}: {e}")
            results[name] = False

    # Verificaciones opcionales
    check_gpu()

    # Resumen final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE CONFIGURACIÓN")
    print("=" * 50)

    all_passed = True
    for name, passed in results.items():
        status = "✅ PASÓ" if passed else "❌ FALLÓ"
        print(f"{name:15} {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)

    if all_passed:
        print("🎉 ¡Sistema configurado correctamente!")
        print("\nPuedes ejecutar la aplicación con:")
        print("   python main.py")
        print("\nO entrenar modelos con:")
        print("   python main.py train")
    else:
        print("⚠️  Algunas verificaciones fallaron")
        print("   Revisa los errores anteriores y vuelve a ejecutar")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
