"""
Main entry point for PoseTrack AI
Real-time movement analysis system

This module serves as the main entry point for the PoseTrack AI application,
providing command-line interface options for running the GUI or training models.
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
    """
    Verify that all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
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
        print("❌ Missing the following dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """
    Create necessary project directories if they don't exist.
    
    Creates models, data, and config directories in the project root.
    """
    directories = [
        project_root / "models",
        project_root / "data",
        project_root / "config"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)

def main():
    """
    Main function that initializes and runs the PoseTrack AI application.
    
    Checks dependencies, creates directories, and launches the GUI interface.
    """
    print("🎯 PoseTrack AI - Movement Analysis System")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies are available")
    
    # Create directories
    create_directories()
    print("✅ Project directories verified")
    
    try:
        # Import and run GUI
        from src.gui.main_gui import main as gui_main
        
        print("🚀 Starting graphical interface...")
        gui_main()
        
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def train_models():
    """
    Function to train specialized models from command line.
    
    Initializes the specialized model trainer and trains all models
    for different activity categories.
    """
    print("🚀 PoseTrack AI - Specialized Model Training")
    print("=" * 50)
    
    if not check_dependencies():
        sys.exit(1)
    
    create_directories()
    
    try:
        from src.training.train_specialized_models import SpecializedModelTrainer
        
        trainer = SpecializedModelTrainer()
        results = trainer.train_all_specialized_models()
        
        print("\n🎉 Specialized model training completed!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "train":
            train_models()
        elif command == "gui":
            main()
        elif command == "--help" or command == "-h":
            print("PoseTrack AI - Usage:")
            print("  python main.py        # Start graphical interface")
            print("  python main.py gui    # Start graphical interface")
            print("  python main.py train  # Train models")
            print("  python main.py --help # Show this help")
        else:
            print(f"Unknown command: {command}")
            print("Use 'python main.py --help' to see available options")
    else:
        # Default: start GUI
        main()
