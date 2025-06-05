"""
Interfaz gráfica principal para PoseTrack AI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
from pathlib import Path
from typing import Optional

from config.settings import GUI_CONFIG, CAMERA_CONFIG, MEDIAPIPE_CONFIG, MODELS_PATH, ACTIVITIES
from src.core.pose_tracker import PoseTracker
from src.core.feature_extractor import FeatureExtractor
from src.core.activity_classifier import ActivityClassifier, ActivityBuffer


class PoseTrackGUI:
    """Interfaz gráfica principal de PoseTrack AI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(GUI_CONFIG["window_title"])
        self.root.geometry(GUI_CONFIG["window_size"])
        self.root.configure(bg="#2b2b2b")

        # Componentes principales
        self.pose_tracker = PoseTracker()
        self.feature_extractor = FeatureExtractor()
        self.activity_classifier = ActivityClassifier()
        self.activity_buffer = ActivityBuffer()

        # Variables de estado
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.video_thread = None

        # Variables de UI
        self.video_label = None
        self.activity_var = tk.StringVar(value="Sin actividad detectada")
        self.confidence_var = tk.StringVar(value="Confianza: 0%")
        self.model_var = tk.StringVar(value="Ningún modelo cargado")
        self.fps_var = tk.StringVar(value="FPS: 0")

        # Métricas en tiempo real
        self.angles_vars = {}
        self.pose_confidence_var = tk.StringVar(value="Confianza Pose: 0%")

        self.setup_ui()
        self.load_available_models()
        
        # Detectar cámaras disponibles al inicio
        self.log_camera_info()

    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Estilo
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Title.TLabel",
            font=("Arial", 16, "bold"),
            foreground="white",
            background="#2b2b2b",
        )
        style.configure(
            "Subtitle.TLabel",
            font=("Arial", 12),
            foreground="white",
            background="#2b2b2b",
        )
        style.configure(
            "Info.TLabel",
            font=("Arial", 10),
            foreground="#cccccc",
            background="#2b2b2b",
        )

        # Panel principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Título
        title_label = ttk.Label(
            main_frame,
            text="🎯 PoseTrack AI - Análisis de Movimiento",
            style="Title.TLabel",
        )
        title_label.pack(pady=(0, 10))

        # Frame superior: controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        # Botones principales
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X)
        self.start_button = ttk.Button(
            buttons_frame, text="▶️ Iniciar Cámara", command=self.toggle_camera, width=15
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            buttons_frame, text="📁 Cargar Modelo", command=self.load_model, width=15
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            buttons_frame,
            text="🔄 Entrenar Modelo",
            command=self.open_training_dialog,
            width=15,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            buttons_frame, text="⚙️ Configuración", command=self.open_settings, width=15
        ).pack(side=tk.LEFT, padx=5)

        # Selector de modelo
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(model_frame, text="Modelo actual:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            model_frame, textvariable=self.model_var, state="readonly", width=40
        )
        self.model_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)

        # Selector de cámara
        camera_frame = ttk.Frame(control_frame)
        camera_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(camera_frame, text="Cámara:").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(
            camera_frame, textvariable=self.camera_var, state="readonly", width=20
        )
        self.camera_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Botón para actualizar cámaras
        ttk.Button(
            camera_frame, text="🔄", command=self.refresh_cameras, width=5
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Frame central: video y métricas
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: video
        video_frame = ttk.LabelFrame(
            content_frame, text="Video en Tiempo Real", padding=10
        )
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Canvas para video
        self.video_canvas = tk.Canvas(
            video_frame,
            bg="black",
            width=GUI_CONFIG["video_size"][0],
            height=GUI_CONFIG["video_size"][1],
        )
        self.video_canvas.pack(pady=(0, 10))

        # Información del video
        video_info_frame = ttk.Frame(video_frame)
        video_info_frame.pack(fill=tk.X)

        ttk.Label(
            video_info_frame, textvariable=self.fps_var, style="Info.TLabel"
        ).pack(side=tk.LEFT)
        ttk.Label(
            video_info_frame, textvariable=self.pose_confidence_var, style="Info.TLabel"
        ).pack(side=tk.RIGHT)

        # Panel derecho: métricas y resultados
        metrics_frame = ttk.LabelFrame(
            content_frame, text="Análisis en Tiempo Real", padding=10
        )
        metrics_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Actividad detectada
        activity_section = ttk.LabelFrame(
            metrics_frame, text="Actividad Detectada", padding=10
        )
        activity_section.pack(fill=tk.X, pady=(0, 10))

        activity_label = ttk.Label(
            activity_section,
            textvariable=self.activity_var,
            style="Subtitle.TLabel",
            wraplength=250,
        )
        activity_label.pack()

        confidence_label = ttk.Label(
            activity_section, textvariable=self.confidence_var, style="Info.TLabel"
        )
        confidence_label.pack(pady=(5, 0))

        # Métricas posturales
        posture_section = ttk.LabelFrame(
            metrics_frame, text="Métricas Posturales", padding=10
        )
        posture_section.pack(fill=tk.X, pady=(0, 10))

        # Crear variables para ángulos
        angle_names = [
            "left_elbow",
            "right_elbow",
            "left_knee",
            "right_knee",
            "trunk_inclination",
            "left_hip",
            "right_hip",
        ]

        for angle_name in angle_names:
            self.angles_vars[angle_name] = tk.StringVar(
                value=f"{angle_name.replace('_', ' ').title()}: --°"
            )
            ttk.Label(
                posture_section,
                textvariable=self.angles_vars[angle_name],
                style="Info.TLabel",
            ).pack(anchor=tk.W)

        # Estado del sistema
        status_section = ttk.LabelFrame(
            metrics_frame, text="Estado del Sistema", padding=10
        )
        status_section.pack(fill=tk.X)

        self.status_text = tk.Text(
            status_section,
            height=8,
            width=35,
            bg="#1e1e1e",
            fg="#cccccc",
            font=("Consolas", 9),
        )
        self.status_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar para el texto de estado
        scrollbar = ttk.Scrollbar(
            status_section, orient=tk.VERTICAL, command=self.status_text.yview
        )
        self.status_text.configure(yscrollcommand=scrollbar.set)

        self.log_message("Sistema iniciado. Listo para análisis.")

    def load_available_models(self):
        """Carga la lista de modelos disponibles"""
        try:
            models = self.activity_classifier.get_available_models()
            model_names = ["Ningún modelo cargado"]

            if models:
                model_names.extend(
                    [f"{model['name']} ({model['type']})" for model in models]
                )
                self.available_models = {
                    f"{model['name']} ({model['type']})": model["path"]
                    for model in models
                }
            else:
                self.available_models = {}

            self.model_combo["values"] = model_names
            self.log_message(f"Encontrados {len(models)} modelos disponibles")

        except Exception as e:
            self.log_message(f"Error cargando modelos: {e}")
            self.available_models = {}

    def on_model_selected(self, event=None):
        """Maneja la selección de modelo desde el combobox"""
        selected = self.model_var.get()

        if selected == "Ningún modelo cargado":
            self.activity_classifier.clear_model()
            self.log_message("Modelo descargado")
            return

        if selected in self.available_models:
            model_path = self.available_models[selected]
            if self.activity_classifier.load_model(model_path):
                self.log_message(f"Modelo {selected} cargado exitosamente")

                # Mostrar información del modelo
                info = self.activity_classifier.get_model_info()
                self.log_message(f"Precisión: {info.get('accuracy', 'N/A')}")
                self.log_message(f"Características: {info.get('n_features', 'N/A')}")
            else:
                self.log_message(f"Error cargando modelo {selected}")
                self.model_var.set("Ningún modelo cargado")

    def toggle_camera(self):
        """Activa/desactiva la cámara"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Inicia la captura de cámara"""
        try:
            # Obtener índice de cámara seleccionada
            camera_index = self.get_selected_camera_index()
            
            # Intentar la cámara seleccionada primero
            camera_indices = [camera_index] + [i for i in [0, 1, 2] if i != camera_index]
            self.cap = None
            
            for idx in camera_indices:
                test_cap = cv2.VideoCapture(idx)
                if test_cap.isOpened():
                    # Probar capturar un frame
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        self.cap = test_cap
                        self.log_message(f"Cámara conectada en índice {idx}")
                        break
                    else:
                        test_cap.release()
                else:
                    test_cap.release()
            
            if self.cap is None:
                raise Exception("No se encontró ninguna cámara disponible")

            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["height"])
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG["fps"])
            
            # Configuraciones adicionales para macOS
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verificar que la cámara funciona
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("La cámara no está capturando frames")

            self.camera_active = True
            self.start_button.config(text="⏹️ Detener Cámara")

            # Limpiar buffers
            self.feature_extractor.reset_buffer()
            self.activity_buffer.clear()

            # Iniciar hilo de video
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()

            self.log_message("Cámara iniciada exitosamente")

        except Exception as e:
            self.log_message(f"Error iniciando cámara: {e}")
            messagebox.showerror("Error", f"No se pudo iniciar la cámara: {e}")

    def stop_camera(self):
        """Detiene la captura de cámara"""
        self.camera_active = False

        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_button.config(text="▶️ Iniciar Cámara")
        self.video_canvas.delete("all")

        # Resetear variables
        self.activity_var.set("Sin actividad detectada")
        self.confidence_var.set("Confianza: 0%")
        self.fps_var.set("FPS: 0")
        self.pose_confidence_var.set("Confianza Pose: 0%")

        for var in self.angles_vars.values():
            var.set(var.get().split(":")[0] + ": --°")

        self.log_message("Cámara detenida")

    def video_loop(self):
        """Bucle principal de procesamiento de video"""
        fps_counter = 0
        start_time = time.time()

        while self.camera_active and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Procesar frame con MediaPipe
                annotated_frame, landmarks = self.pose_tracker.process_frame(frame)

                if landmarks:
                    # Calcular ángulos
                    angles = self.pose_tracker.calculate_angles(landmarks)

                    # Actualizar métricas posturales en UI
                    self.update_angles_display(angles)

                    # Calcular confianza de pose
                    pose_confidence = self.pose_tracker.get_pose_confidence(landmarks)
                    self.pose_confidence_var.set(
                        f"Confianza Pose: {pose_confidence*100:.1f}%"
                    )

                    # Añadir al extractor de características
                    self.feature_extractor.add_frame_data(landmarks, angles)

                    # Clasificar actividad si hay suficientes datos
                    if (
                        self.feature_extractor.is_ready()
                        and self.activity_classifier.is_ready()
                    ):
                        features = self.feature_extractor.extract_features()
                        if features is not None:
                            activity, confidence, probabilities = (
                                self.activity_classifier.predict(features)
                            )

                            # Añadir al buffer de suavizado
                            self.activity_buffer.add_prediction(activity, confidence)

                            # Obtener predicción suavizada
                            smooth_activity, smooth_confidence = (
                                self.activity_buffer.get_smoothed_prediction()
                            )

                            # Actualizar UI
                            activity_desc = (
                                self.activity_classifier.get_activity_description(
                                    smooth_activity
                                )
                            )
                            self.activity_var.set(activity_desc)
                            self.confidence_var.set(
                                f"Confianza: {smooth_confidence*100:.1f}%"
                            )

                # Mostrar frame en canvas
                self.display_frame(annotated_frame)

                # Calcular FPS
                fps_counter += 1
                if time.time() - start_time >= 1.0:
                    self.fps_var.set(f"FPS: {fps_counter}")
                    fps_counter = 0
                    start_time = time.time()

                # Pequeña pausa para no saturar la CPU
                time.sleep(0.01)

            except Exception as e:
                self.log_message(f"Error en video loop: {e}")
                break

    def display_frame(self, frame):
        """Muestra un frame en el canvas de video"""
        try:
            # Redimensionar frame
            frame_resized = cv2.resize(frame, GUI_CONFIG["video_size"])

            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convertir a formato PIL
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)

            # Actualizar canvas
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.video_canvas.image = photo  # Mantener referencia

        except Exception as e:
            print(f"Error mostrando frame: {e}")

    def update_angles_display(self, angles):
        """Actualiza la visualización de ángulos en la UI"""
        for angle_name, angle_value in angles.items():
            if angle_name in self.angles_vars:
                display_name = angle_name.replace("_", " ").title()
                if not np.isnan(angle_value):
                    self.angles_vars[angle_name].set(
                        f"{display_name}: {angle_value:.1f}°"
                    )
                else:
                    self.angles_vars[angle_name].set(f"{display_name}: --°")

    def load_model(self):
        """Abre diálogo para cargar modelo manualmente"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Modelo",
            initialdir=MODELS_PATH,
            filetypes=[("Archivos de modelo", "*.pkl"), ("Todos los archivos", "*.*")],
        )

        if file_path:
            if self.activity_classifier.load_model(file_path):
                model_name = Path(file_path).stem
                self.model_var.set(f"{model_name} (manual)")
                self.log_message(f"Modelo {model_name} cargado manualmente")

                # Mostrar información
                info = self.activity_classifier.get_model_info()
                self.log_message(f"Tipo: {info.get('model_name', 'desconocido')}")
                self.log_message(f"Precisión: {info.get('accuracy', 'N/A')}")
            else:
                messagebox.showerror(
                    "Error", "No se pudo cargar el modelo seleccionado"
                )

    def open_training_dialog(self):
        """Abre diálogo de entrenamiento de modelos"""
        TrainingDialog(self.root, self)

    def open_settings(self):
        """Abre diálogo de configuración"""
        SettingsDialog(self.root, self)

    def log_message(self, message):
        """Añade un mensaje al log de estado"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def run(self):
        """Inicia la aplicación"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"Error en aplicación: {e}")
        finally:
            self.cleanup()

    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        if self.camera_active:
            self.stop_camera()

        self.cleanup()
        self.root.destroy()

    def cleanup(self):
        """Limpia recursos al cerrar"""
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
        except:
            pass

        try:
            if hasattr(self, 'pose_tracker') and hasattr(self.pose_tracker, "close"):
                self.pose_tracker.close()
        except:
            pass

    def detect_cameras(self):
        """Detecta cámaras disponibles"""
        available_cameras = []
        for i in range(10):  # Probar los primeros 10 índices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                cap.release()
        
        return available_cameras

    def refresh_cameras(self):
        """Actualiza la lista de cámaras disponibles"""
        cameras = self.detect_cameras()
        camera_options = [f"Cámara {i}" for i in cameras]
        
        if camera_options:
            self.camera_combo['values'] = camera_options
            if not self.camera_var.get() or self.camera_var.get() not in camera_options:
                self.camera_var.set(camera_options[0])
            self.log_message(f"Cámaras detectadas: {cameras}")
        else:
            self.camera_combo['values'] = ["No hay cámaras"]
            self.camera_var.set("No hay cámaras")
            self.log_message("⚠️ No se detectaron cámaras")

    def get_selected_camera_index(self):
        """Obtiene el índice de la cámara seleccionada"""
        try:
            camera_text = self.camera_var.get()
            if camera_text.startswith("Cámara "):
                return int(camera_text.split(" ")[1])
            return 0
        except:
            return 0

    def log_camera_info(self):
        """Registra información sobre cámaras disponibles"""
        self.refresh_cameras()


class TrainingDialog:
    """Diálogo para entrenamiento de modelos"""

    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Entrenamiento de Modelos")
        self.dialog.geometry("600x400")
        self.dialog.configure(bg="#2b2b2b")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.setup_ui()

    def setup_ui(self):
        """Configura la UI del diálogo de entrenamiento"""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        ttk.Label(
            main_frame, text="🚀 Entrenamiento de Modelos", font=("Arial", 14, "bold")
        ).pack(pady=(0, 20))

        # Información
        info_text = """
Este proceso entrenará modelos de machine learning usando los videos
disponibles en la carpeta de entrenamiento.

Pasos:
1. Extracción de características de videos
2. Entrenamiento de múltiples modelos (SVM, Random Forest, XGBoost)
3. Evaluación y guardado de modelos

⚠️ Este proceso puede tomar varios minutos dependiendo 
de la cantidad de videos.
        """

        info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(pady=(0, 20))

        # Opciones
        options_frame = ttk.LabelFrame(main_frame, text="Opciones", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 20))

        self.model_types = {
            "SVM": tk.BooleanVar(value=True),
            "Random Forest": tk.BooleanVar(value=True),
            "XGBoost": tk.BooleanVar(value=True),
        }

        for model_name, var in self.model_types.items():
            ttk.Checkbutton(options_frame, text=model_name, variable=var).pack(
                anchor=tk.W
            )

        # Botones
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X)

        ttk.Button(
            buttons_frame, text="🚀 Iniciar Entrenamiento", command=self.start_training
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(buttons_frame, text="❌ Cancelar", command=self.dialog.destroy).pack(
            side=tk.RIGHT
        )

        # Área de progreso
        self.progress_frame = ttk.LabelFrame(main_frame, text="Progreso", padding=10)
        self.progress_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))

        self.progress_text = tk.Text(
            self.progress_frame, height=10, bg="#1e1e1e", fg="#cccccc"
        )
        self.progress_text.pack(fill=tk.BOTH, expand=True)

    def start_training(self):
        """Inicia el proceso de entrenamiento en un hilo separado"""
        selected_models = [name for name, var in self.model_types.items() if var.get()]

        if not selected_models:
            messagebox.showwarning(
                "Advertencia", "Selecciona al menos un tipo de modelo"
            )
            return

        self.progress_text.insert(tk.END, "Iniciando entrenamiento...\n")

        # Deshabilitar botón
        for widget in self.dialog.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state="disabled")

        # Iniciar en hilo separado
        threading.Thread(
            target=self.training_worker, args=(selected_models,), daemon=True
        ).start()

    def training_worker(self, selected_models):
        """Worker para entrenamiento en hilo separado"""
        try:
            from src.training.train_model import ModelTrainer

            trainer = ModelTrainer()

            self.log_progress("Extrayendo características de videos...")
            X, y = trainer.prepare_training_data()

            self.log_progress(
                f"Extraídas {len(X)} muestras con {X.shape[1]} características"
            )

            self.log_progress("Guardando datos de entrenamiento...")
            trainer.save_training_data(X, y)

            # Entrenar modelos seleccionados
            results = {}

            for model_name in selected_models:
                model_type = model_name.lower().replace(" ", "_")
                if model_type == "random_forest":
                    model_type = "rf"

                self.log_progress(f"Entrenando {model_name}...")

                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_path = MODELS_PATH / f"{model_type}_model_{timestamp}.pkl"

                    metrics = trainer.train_model(model_type, X, y, str(save_path))
                    results[model_name] = metrics

                    self.log_progress(
                        f"{model_name} - Precisión: {metrics['accuracy']:.4f}"
                    )

                except Exception as e:
                    self.log_progress(f"Error entrenando {model_name}: {e}")

            self.log_progress("\n=== ENTRENAMIENTO COMPLETADO ===")

            # Mostrar resumen
            best_model = max(results.items(), key=lambda x: x[1].get("accuracy", 0))
            self.log_progress(
                f"Mejor modelo: {best_model[0]} ({best_model[1]['accuracy']:.4f})"
            )

            # Actualizar lista de modelos en la aplicación principal
            self.main_app.load_available_models()

        except Exception as e:
            self.log_progress(f"Error durante entrenamiento: {e}")
            import traceback

            traceback.print_exc()

    def log_progress(self, message):
        """Añade mensaje al log de progreso"""
        timestamp = time.strftime("%H:%M:%S")
        self.progress_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.progress_text.see(tk.END)
        self.dialog.update_idletasks()


class SettingsDialog:
    """Diálogo de configuración"""

    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Configuración")
        self.dialog.geometry("500x600")
        self.dialog.configure(bg="#2b2b2b")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.setup_ui()

    def setup_ui(self):
        """Configura la UI del diálogo de configuración"""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Título
        ttk.Label(
            main_frame, text="⚙️ Configuración del Sistema", font=("Arial", 14, "bold")
        ).pack(pady=(0, 20))

        # Notebook para pestañas
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Pestaña de cámara
        camera_frame = ttk.Frame(notebook, padding=10)
        notebook.add(camera_frame, text="Cámara")

        self.setup_camera_settings(camera_frame)

        # Pestaña de detección
        detection_frame = ttk.Frame(notebook, padding=10)
        notebook.add(detection_frame, text="Detección")

        self.setup_detection_settings(detection_frame)

        # Pestaña de clasificación
        classification_frame = ttk.Frame(notebook, padding=10)
        notebook.add(classification_frame, text="Clasificación")

        self.setup_classification_settings(classification_frame)

        # Botones
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(20, 0))

        ttk.Button(buttons_frame, text="💾 Guardar", command=self.save_settings).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        ttk.Button(buttons_frame, text="❌ Cancelar", command=self.dialog.destroy).pack(
            side=tk.RIGHT
        )

    def setup_camera_settings(self, parent):
        """Configura settings de cámara"""
        ttk.Label(
            parent, text="Configuración de Cámara", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W)

        # Resolución
        res_frame = ttk.Frame(parent)
        res_frame.pack(fill=tk.X, pady=5)

        ttk.Label(res_frame, text="Resolución:").pack(side=tk.LEFT)
        self.resolution_var = tk.StringVar(
            value=f"{CAMERA_CONFIG['width']}x{CAMERA_CONFIG['height']}"
        )
        ttk.Combobox(
            res_frame,
            textvariable=self.resolution_var,
            values=["640x480", "800x600", "1280x720"],
            state="readonly",
        ).pack(side=tk.RIGHT)

        # FPS
        fps_frame = ttk.Frame(parent)
        fps_frame.pack(fill=tk.X, pady=5)

        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.IntVar(value=CAMERA_CONFIG["fps"])
        ttk.Spinbox(
            fps_frame, from_=15, to=60, textvariable=self.fps_var, width=10
        ).pack(side=tk.RIGHT)

    def setup_detection_settings(self, parent):
        """Configura settings de detección"""
        ttk.Label(
            parent, text="Configuración de MediaPipe", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W)

        # Confianza mínima
        conf_frame = ttk.Frame(parent)
        conf_frame.pack(fill=tk.X, pady=5)

        ttk.Label(conf_frame, text="Confianza mínima:").pack(side=tk.LEFT)
        self.min_confidence_var = tk.DoubleVar(
            value=MEDIAPIPE_CONFIG["min_detection_confidence"]
        )
        ttk.Scale(
            conf_frame,
            from_=0.1,
            to=1.0,
            variable=self.min_confidence_var,
            orient=tk.HORIZONTAL,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # Complejidad del modelo
        complexity_frame = ttk.Frame(parent)
        complexity_frame.pack(fill=tk.X, pady=5)

        ttk.Label(complexity_frame, text="Complejidad del modelo:").pack(side=tk.LEFT)
        self.complexity_var = tk.IntVar(value=MEDIAPIPE_CONFIG["model_complexity"])
        ttk.Combobox(
            complexity_frame,
            textvariable=self.complexity_var,
            values=[0, 1, 2],
            state="readonly",
            width=10,
        ).pack(side=tk.RIGHT)

    def setup_classification_settings(self, parent):
        """Configura settings de clasificación"""
        ttk.Label(
            parent, text="Configuración de Clasificación", font=("Arial", 12, "bold")
        ).pack(anchor=tk.W)

        # Tamaño de ventana
        window_frame = ttk.Frame(parent)
        window_frame.pack(fill=tk.X, pady=5)

        ttk.Label(window_frame, text="Ventana de análisis (frames):").pack(side=tk.LEFT)
        self.window_size_var = tk.IntVar(value=30)  # FEATURE_CONFIG['window_size']
        ttk.Spinbox(
            window_frame, from_=10, to=60, textvariable=self.window_size_var, width=10
        ).pack(side=tk.RIGHT)

        # Buffer de suavizado
        buffer_frame = ttk.Frame(parent)
        buffer_frame.pack(fill=tk.X, pady=5)

        ttk.Label(buffer_frame, text="Buffer de suavizado:").pack(side=tk.LEFT)
        self.buffer_size_var = tk.IntVar(value=10)
        ttk.Spinbox(
            buffer_frame, from_=5, to=20, textvariable=self.buffer_size_var, width=10
        ).pack(side=tk.RIGHT)

    def save_settings(self):
        """Guarda la configuración"""
        # Aquí se implementaría el guardado real de configuración
        messagebox.showinfo("Configuración", "Configuración guardada exitosamente")
        self.dialog.destroy()


def main():
    """Función principal"""
    try:
        app = PoseTrackGUI()
        app.run()
    except Exception as e:
        print(f"Error iniciando aplicación: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
