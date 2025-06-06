"""
Archivo principal de PoseTrack AI
Sistema de análisis de movimiento en tiempo real
"""

import sys
import os
from pathlib import Path

# Añadir el directorio src al path de Python
project_root = Path(__file__).parent
src_path = project_root / "src"
config_path = project_root / "config"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(config_path))

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    required_packages = [
        'cv2', 'mediapipe', 'numpy', 'pandas', 'sklearn', 
        'xgboost', 'joblib', 'tkinter', 'PIL', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Faltan las siguientes dependencias:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstálalas con: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Crea los directorios necesarios si no existen"""
    directories = [
        project_root / "models",
        project_root / "data",
        project_root / "config"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)

def main():
    """Función principal"""
    print("🎯 PoseTrack AI - Sistema de Análisis de Movimiento")
    print("=" * 50)
    
    # Verificar dependencias
    print("Verificando dependencias...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ Todas las dependencias están disponibles")
    
    # Crear directorios
    create_directories()
    print("✅ Directorios del proyecto verificados")
    
    try:
        # Importar y ejecutar la GUI
        from src.gui.main_gui import main as gui_main
        
        print("🚀 Iniciando interfaz gráfica...")
        gui_main()
        
    except KeyboardInterrupt:
        print("\n👋 Aplicación cerrada por el usuario")
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def train_models():
    """Función para entrenar modelos desde línea de comandos"""
    print("🚀 PoseTrack AI - Entrenamiento de Modelos")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    create_directories()
    
    try:
        from src.training.train_model import main as train_main
        train_main()
    except Exception as e:
        print(f"❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            train_models()
        elif command == "gui":
            main()
        elif command == "--help" or command == "-h":
            print("PoseTrack AI - Uso:")
            print("  python main.py        # Iniciar interfaz gráfica")
            print("  python main.py gui    # Iniciar interfaz gráfica")
            print("  python main.py train  # Entrenar modelos")
            print("  python main.py --help # Mostrar esta ayuda")
        else:
            print(f"Comando desconocido: {command}")
            print("Usa 'python main.py --help' para ver opciones disponibles")
    else:
        # Por defecto, iniciar GUI
        main()
