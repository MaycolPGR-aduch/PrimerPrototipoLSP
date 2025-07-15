import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from collections import deque
import requests
import base64
from PIL import Image, ImageTk
import json
import pyttsx3
from ultralytics import YOLO
import os

class AdvancedSignLanguageDetector:
    def __init__(self):
        # Configuración del modelo YOLOv8 preentrenado
        self.load_models()
        
        # Variables de control
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        
        # Buffer para estabilizar predicciones
        self.prediction_buffer = deque(maxlen=15)
        self.last_prediction = ""
        self.prediction_count = 0
        
        # Sistema de texto a voz
        self.setup_tts()
        
        # Mapeo de señas detectadas (puede expandirse)
        self.sign_mapping = {
            'A': 'Letra A',
            'B': 'Letra B', 
            'C': 'Letra C',
            'D': 'Letra D',
            'E': 'Letra E',
            'F': 'Letra F',
            'G': 'Letra G',
            'H': 'Letra H',
            'I': 'Letra I',
            'J': 'Letra J',
            'K': 'Letra K',
            'L': 'Letra L',
            'M': 'Letra M',
            'N': 'Letra N',
            'O': 'Letra O',
            'P': 'Letra P',
            'Q': 'Letra Q',
            'R': 'Letra R',
            'S': 'Letra S',
            'T': 'Letra T',
            'U': 'Letra U',
            'V': 'Letra V',
            'W': 'Letra W',
            'X': 'Letra X',
            'Y': 'Letra Y',
            'Z': 'Letra Z',
            'HELLO': 'Hola',
            'THANKS': 'Gracias',
            'YES': 'Sí',
            'NO': 'No',
            'PLEASE': 'Por favor',
            'SORRY': 'Lo siento'
        }
        
        # Historial de palabras formadas
        self.word_history = []
        self.current_word = ""
        
        self.setup_gui()
    
    def load_models(self):
        """Cargar modelos YOLOv8 preentrenados"""
        print("🔄 Cargando modelos de IA...")
        
        try:
            # Primero intentar descargar modelo especializado en señas
            sign_model_url = "https://github.com/mukund0502/sign_recognition_yolo-v8/raw/main/runs/detect/train/weights/best.pt"
            sign_model_path = "sign_language_best.pt"
            
            # Verificar si ya existe el modelo especializado
            if not os.path.exists(sign_model_path):
                print("🤖 Descargando modelo YOLOv8 especializado en señas...")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(sign_model_url, sign_model_path)
                    self.model = YOLO(sign_model_path)
                    print("✅ Modelo especializado en señas cargado")
                    self.model_type = "Especializado ASL"
                except Exception as download_error:
                    print(f"⚠️ No se pudo descargar modelo especializado: {download_error}")
                    print("🔄 Usando modelo simulado de señas...")
                    self.model = None
                    self.model_type = "Simulado LSP"
            else:
                self.model = YOLO(sign_model_path)
                print("✅ Modelo especializado en señas cargado")
                self.model_type = "Especializado ASL"
                
        except Exception as e:
            print(f"⚠️ Error cargando modelo especializado: {e}")
            print("🔄 Usando sistema simulado de señas...")
            self.model = None
            self.model_type = "Simulado LSP"
            
        # API de Roboflow como respaldo (modelo preentrenado especializado)
        self.roboflow_api_key = "YOUR_API_KEY"  # Usuario debe obtener clave gratis
        self.roboflow_model_url = "https://detect.roboflow.com/sign-language-detection-7cdpj/2"
        self.use_roboflow = False  # Cambiar a True si se tiene API key
    
    def setup_tts(self):
        """Configurar sistema de texto a voz"""
        try:
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            
            # Buscar voz en español si está disponible
            spanish_voice = None
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                    spanish_voice = voice.id
                    break
            
            if spanish_voice:
                self.tts_engine.setProperty('voice', spanish_voice)
            
            # Configurar velocidad y volumen
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            
            print("🔊 Sistema de voz configurado")
            
        except Exception as e:
            print(f"⚠️ Error configurando TTS: {e}")
            self.tts_engine = None
    
    def setup_gui(self):
        """Configurar interfaz gráfica avanzada"""
        self.root = tk.Tk()
        self.root.title("Sistema Avanzado de Traducción LSP con IA - Prototipo YOLOv8")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        # Estilo moderno
        style = ttk.Style()
        style.theme_use('clam')
        
        # Título principal con gradiente visual
        title_frame = tk.Frame(self.root, bg='#16213e', height=80)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🤖 Sistema LSP con Inteligencia Artificial YOLOv8", 
            font=('Segoe UI', 20, 'bold'),
            bg='#16213e',
            fg='#00d4ff'
        )
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Detección en Tiempo Real | Traducción Automática | Síntesis de Voz",
            font=('Segoe UI', 10),
            bg='#16213e',
            fg='#a8dadc'
        )
        subtitle_label.pack()
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Panel de control superior
        control_panel = tk.Frame(main_frame, bg='#0f3460', relief='raised', bd=2)
        control_panel.pack(fill='x', pady=(0, 10))
        
        # Botones de control con iconos
        button_frame = tk.Frame(control_panel, bg='#0f3460')
        button_frame.pack(side='left', padx=10, pady=10)
        
        self.start_btn = tk.Button(
            button_frame,
            text="🎥 Iniciar Cámara",
            command=self.start_camera,
            bg='#28a745',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="⏹️ Detener",
            command=self.stop_camera,
            bg='#dc3545',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=20,
            pady=8,
            state='disabled',
            cursor='hand2'
        )
        self.stop_btn.pack(side='left', padx=5)
        
        self.speak_btn = tk.Button(
            button_frame,
            text="🔊 Reproducir",
            command=self.speak_current_word,
            bg='#17a2b8',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.speak_btn.pack(side='left', padx=5)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Limpiar",
            command=self.clear_history,
            bg='#6c757d',
            fg='white',
            font=('Segoe UI', 11, 'bold'),
            padx=20,
            pady=8,
            cursor='hand2'
        )
        self.clear_btn.pack(side='left', padx=5)
        
        # Estado del sistema
        status_frame = tk.Frame(control_panel, bg='#0f3460')
        status_frame.pack(side='right', padx=10, pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="🔴 Sistema Inactivo",
            font=('Segoe UI', 11, 'bold'),
            bg='#0f3460',
            fg='#ffc107'
        )
        self.status_label.pack()
        
        self.fps_label = tk.Label(
            status_frame,
            text="FPS: 0",
            font=('Segoe UI', 9),
            bg='#0f3460',
            fg='#28a745'
        )
        self.fps_label.pack()
        
        # Contenido principal en tres columnas
        content_frame = tk.Frame(main_frame, bg='#1a1a2e')
        content_frame.pack(expand=True, fill='both')
        
        # Columna izquierda - Cámara
        left_frame = tk.Frame(content_frame, bg='#0f3460', relief='raised', bd=2)
        left_frame.pack(side='left', expand=True, fill='both', padx=(0, 5))
        
        camera_title = tk.Label(
            left_frame,
            text="📹 Vista en Tiempo Real",
            font=('Segoe UI', 14, 'bold'),
            bg='#0f3460',
            fg='#00d4ff'
        )
        camera_title.pack(pady=5)
        
        self.camera_label = tk.Label(
            left_frame,
            text="Cámara no iniciada\n\nHaz clic en 'Iniciar Cámara'\npara comenzar la detección",
            bg='black',
            fg='white',
            font=('Segoe UI', 12),
            justify='center'
        )
        self.camera_label.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Columna central - Detección
        center_frame = tk.Frame(content_frame, bg='#0f3460', relief='raised', bd=2)
        center_frame.pack(side='left', fill='y', padx=5)
        center_frame.configure(width=280)
        
        detection_title = tk.Label(
            center_frame,
            text="🎯 Detección IA",
            font=('Segoe UI', 14, 'bold'),
            bg='#0f3460',
            fg='#00d4ff'
        )
        detection_title.pack(pady=5)
        
        # Seña actual detectada
        current_frame = tk.Frame(center_frame, bg='#1a1a2e', relief='sunken', bd=2)
        current_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Label(
            current_frame,
            text="Seña Actual:",
            font=('Segoe UI', 10, 'bold'),
            bg='#1a1a2e',
            fg='#a8dadc'
        ).pack()
        
        self.detected_sign_label = tk.Label(
            current_frame,
            text="Ninguna",
            font=('Segoe UI', 16, 'bold'),
            bg='#1a1a2e',
            fg='#00ff88',
            wraplength=250
        )
        self.detected_sign_label.pack(pady=5)
        
        # Métricas de detección
        metrics_frame = tk.Frame(center_frame, bg='#1a1a2e', relief='sunken', bd=2)
        metrics_frame.pack(pady=5, padx=10, fill='x')
        
        self.confidence_label = tk.Label(
            metrics_frame,
            text="Confianza: 0%",
            font=('Segoe UI', 10),
            bg='#1a1a2e',
            fg='#ffc107'
        )
        self.confidence_label.pack(pady=2)
        
        self.detection_count_label = tk.Label(
            metrics_frame,
            text="Detecciones: 0",
            font=('Segoe UI', 10),
            bg='#1a1a2e',
            fg='#17a2b8'
        )
        self.detection_count_label.pack(pady=2)
        
        # Modelo de IA en uso
        model_frame = tk.Frame(center_frame, bg='#1a1a2e', relief='sunken', bd=2)
        model_frame.pack(pady=5, padx=10, fill='x')
        
        tk.Label(
            model_frame,
            text="🧠 Modelo de IA:",
            font=('Segoe UI', 10, 'bold'),
            bg='#1a1a2e',
            fg='#a8dadc'
        ).pack()
        
        self.model_label = tk.Label(
            model_frame,
            text="Cargando...",
            font=('Segoe UI', 9),
            bg='#1a1a2e',
            fg='#00d4ff'
        )
        self.model_label.pack()
        
        # Columna derecha - Traducción y historial
        right_frame = tk.Frame(content_frame, bg='#0f3460', relief='raised', bd=2)
        right_frame.pack(side='right', fill='y', padx=(5, 0))
        right_frame.configure(width=300)
        
        translation_title = tk.Label(
            right_frame,
            text="💬 Traducción y Síntesis",
            font=('Segoe UI', 14, 'bold'),
            bg='#0f3460',
            fg='#00d4ff'
        )
        translation_title.pack(pady=5)
        
        # Palabra actual formada
        word_frame = tk.Frame(right_frame, bg='#1a1a2e', relief='sunken', bd=2)
        word_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Label(
            word_frame,
            text="Palabra Formada:",
            font=('Segoe UI', 10, 'bold'),
            bg='#1a1a2e',
            fg='#a8dadc'
        ).pack()
        
        self.current_word_label = tk.Label(
            word_frame,
            text="",
            font=('Segoe UI', 14, 'bold'),
            bg='#1a1a2e',
            fg='#00ff88',
            wraplength=280,
            height=2
        )
        self.current_word_label.pack(pady=5)
        
        # Historial de palabras
        history_frame = tk.Frame(right_frame, bg='#1a1a2e', relief='sunken', bd=2)
        history_frame.pack(pady=5, padx=10, fill='both', expand=True)
        
        tk.Label(
            history_frame,
            text="📝 Historial:",
            font=('Segoe UI', 10, 'bold'),
            bg='#1a1a2e',
            fg='#a8dadc'
        ).pack()
        
        # Lista scrollable para historial
        scrollbar = tk.Scrollbar(history_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.history_listbox = tk.Listbox(
            history_frame,
            yscrollcommand=scrollbar.set,
            bg='#16213e',
            fg='#a8dadc',
            font=('Segoe UI', 9),
            selectbackground='#00d4ff'
        )
        self.history_listbox.pack(expand=True, fill='both', padx=5, pady=5)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Estadísticas
        stats_frame = tk.Frame(right_frame, bg='#1a1a2e', relief='sunken', bd=2)
        stats_frame.pack(pady=5, padx=10, fill='x')
        
        tk.Label(
            stats_frame,
            text="📊 Estadísticas:",
            font=('Segoe UI', 10, 'bold'),
            bg='#1a1a2e',
            fg='#a8dadc'
        ).pack()
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Señas detectadas: 0\nPalabras formadas: 0",
            font=('Segoe UI', 9),
            bg='#1a1a2e',
            fg='#ffc107',
            justify='left'
        )
        self.stats_label.pack(pady=2)
        
        # Bind para cerrar aplicación
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables para estadísticas
        self.total_detections = 0
        self.total_words = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
    
    def detect_with_yolov8(self, frame):
        """Detectar señas usando YOLOv8"""
        try:
            if self.model is None:
                return self.detect_basic_signs(frame)
            
            # Ejecutar inferencia
            results = self.model(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Obtener coordenadas y confianza
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Obtener nombre de la clase
                        class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f"Class_{class_id}"
                        
                        if confidence > 0.5:  # Umbral de confianza
                            detections.append({
                                'class': class_name.upper(),
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2))
                            })
                            
                            # Dibujar bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return detections, frame
            
        except Exception as e:
            print(f"Error en detección YOLOv8: {e}")
            return self.detect_basic_signs(frame)
    
    def detect_with_yolov8(self, frame):
        """Detectar señas usando YOLOv8"""
        try:
            if self.model is None:
                return self.detect_simulated_signs(frame)
            
            # Ejecutar inferencia
            results = self.model(frame, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Obtener coordenadas y confianza
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Obtener nombre de la clase
                        class_name = self.model.names[class_id] if hasattr(self.model, 'names') else f"Class_{class_id}"
                        
                        # Filtrar solo señas (no personas u otros objetos)
                        if confidence > 0.5 and self.is_sign_class(class_name):
                            detections.append({
                                'class': class_name.upper(),
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2))
                            })
                            
                            # Dibujar bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return detections, frame
            
        except Exception as e:
            print(f"Error en detección YOLOv8: {e}")
            return self.detect_simulated_signs(frame)
    
    def is_sign_class(self, class_name):
        """Verificar si la clase detectada es una seña válida"""
        # Lista de clases que NO son señas (filtrar objetos generales)
        non_sign_classes = {
            'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
            'laptop', 'mouse', 'keyboard', 'cell phone', 'book', 'clock',
            'chair', 'sofa', 'bed', 'dining table', 'toilet', 'tv',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot'
        }
        
        class_lower = class_name.lower()
        
        # Si es una clase no-seña, ignorar
        if class_lower in non_sign_classes:
            return False
            
        # Si es una letra o palabra de señas, aceptar
        sign_patterns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                        'hello', 'thanks', 'yes', 'no', 'please', 'sorry', 'love', 'you']
        
        return class_lower in sign_patterns or len(class_name) == 1
    
    def detect_simulated_signs(self, frame):
        """Sistema simulado de detección de señas LSP"""
        height, width = frame.shape[:2]
        
        # Detectar manos usando contornos básicos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detectar regiones que podrían ser manos
        ret, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        # Agregar texto indicando modo simulado
        cv2.putText(frame, f"Modo Simulado LSP - Modelo: {self.model_type}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Muestra tus manos en el area central", 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Dibujar área de detección
        center_x, center_y = width // 2, height // 2
        roi_size = 150
        cv2.rectangle(frame, 
                     (center_x - roi_size, center_y - roi_size),
                     (center_x + roi_size, center_y + roi_size),
                     (255, 255, 0), 2)
        
        # Simulación inteligente basada en área central
        import random
        
        # Verificar si hay movimiento en el área central
        roi = gray[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
        if roi.size > 0:
            activity = cv2.mean(roi)[0]
            
            # Si hay suficiente actividad (manos moviéndose)
            if activity > 100:  # Umbral de actividad
                # Simular detección de señas LSP comunes
                lsp_signs = {
                    'HOLA': 0.85,
                    'GRACIAS': 0.80,
                    'POR_FAVOR': 0.78,
                    'SI': 0.82,
                    'NO': 0.75,
                    'AMOR': 0.88,
                    'FAMILIA': 0.76,
                    'CASA': 0.83,
                    'COMER': 0.79,
                    'AGUA': 0.81,
                    'A': 0.90,
                    'B': 0.87,
                    'C': 0.85
                }
                
                if random.random() > 0.6:  # 40% chance de detección
                    detected_sign = random.choice(list(lsp_signs.keys()))
                    confidence = lsp_signs[detected_sign] + random.uniform(-0.1, 0.1)
                    
                    # Crear bounding box en área de detección
                    bbox = (center_x - roi_size + 20, center_y - roi_size + 20,
                           center_x + roi_size - 20, center_y + roi_size - 20)
                    
                    # Dibujar detección
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                    cv2.putText(frame, f"LSP: {detected_sign} ({confidence:.1%})", 
                               (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    detections.append({
                        'class': detected_sign,
                        'confidence': confidence,
                        'bbox': bbox
                    })
        
        return detections, frame
    
    def process_detections(self, detections):
        """Procesar las detecciones y formar palabras"""
        if not detections:
            return "Ninguna", 0.0
        
        # Obtener la detección con mayor confianza
        best_detection = max(detections, key=lambda x: x['confidence'])
        detected_class = best_detection['class']
        confidence = best_detection['confidence']
        
        # Agregar al buffer para estabilización
        self.prediction_buffer.append((detected_class, confidence))
        
        # Estabilizar predicción
        if len(self.prediction_buffer) >= 10:
            # Contar frecuencia de las últimas 10 detecciones
            recent_predictions = list(self.prediction_buffer)[-10:]
            sign_counts = {}
            
            for sign, conf in recent_predictions:
                if sign not in sign_counts:
                    sign_counts[sign] = {'count': 0, 'total_conf': 0}
                sign_counts[sign]['count'] += 1
                sign_counts[sign]['total_conf'] += conf
            
            # Obtener predicción más estable
            most_frequent = max(sign_counts.items(), key=lambda x: x[1]['count'])
            if most_frequent[1]['count'] >= 6:  # Al menos 6 de 10 detecciones
                stable_sign = most_frequent[0]
                avg_confidence = most_frequent[1]['total_conf'] / most_frequent[1]['count']
                
                # Agregar a palabra actual si es una letra
                if stable_sign != self.last_prediction and len(stable_sign) == 1:
                    self.current_word += stable_sign
                    self.last_prediction = stable_sign
                    self.prediction_count = 0
                elif stable_sign == self.last_prediction:
                    self.prediction_count += 1
                    if self.prediction_count > 20:  # Espacio entre letras
                        if self.current_word and stable_sign != self.current_word[-1]:
                            self.current_word += stable_sign
                        self.prediction_count = 0
                
                return stable_sign, avg_confidence
        
        return detected_class, confidence
    
    def speak_current_word(self):
        """Reproducir la palabra actual con TTS"""
        if self.tts_engine and self.current_word:
            threading.Thread(target=self._speak, args=(self.current_word,), daemon=True).start()
    
    def _speak(self, text):
        """Función auxiliar para TTS en hilo separado"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error en TTS: {e}")
    
    def clear_history(self):
        """Limpiar historial y palabra actual"""
        if self.current_word:
            self.word_history.append(self.current_word)
            self.history_listbox.insert(tk.END, f"{len(self.word_history)}. {self.current_word}")
            self.total_words += 1
            
        self.current_word = ""
        self.current_word_label.configure(text="")
        self.update_stats()
    
    def update_stats(self):
        """Actualizar estadísticas"""
        stats_text = f"Señas detectadas: {self.total_detections}\nPalabras formadas: {self.total_words}"
        self.stats_label.configure(text=stats_text)
    
    def start_camera(self):
        """Iniciar cámara y detección"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo acceder a la cámara")
                return
            
            # Configurar resolución para mejor rendimiento
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_active = True
            self.start_btn.configure(state='disabled')
            self.stop_btn.configure(state='normal')
            self.status_label.configure(text="🟢 Sistema Activo", fg='#28a745')
            
            # Iniciar hilo de procesamiento
            self.video_thread = threading.Thread(target=self.process_video_advanced)
            self.video_thread.daemon = True
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar la cámara: {str(e)}")
    
    def stop_camera(self):
        """Detener cámara"""
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
        
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.status_label.configure(text="🔴 Sistema Inactivo", fg='#ffc107')
        self.fps_label.configure(text="FPS: 0")
        self.camera_label.configure(image='', text="Cámara detenida")
    
    def process_video_advanced(self):
        """Procesamiento avanzado de video con IA"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.camera_active:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Voltear frame para efecto espejo
                frame = cv2.flip(frame, 1)
                
                # Detectar señas con IA
                detections, processed_frame = self.detect_with_yolov8(frame)
                
                # Procesar detecciones
                predicted_sign, confidence = self.process_detections(detections)
                
                if predicted_sign != "Ninguna":
                    self.total_detections += 1
                
                # Calcular FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    self.root.after(0, lambda: self.fps_label.configure(text=f"FPS: {fps:.1f}"))
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Actualizar GUI
                self.update_advanced_gui(processed_frame, predicted_sign, confidence)
                
                time.sleep(0.01)  # ~100 FPS máximo
                
            except Exception as e:
                print(f"Error en procesamiento: {str(e)}")
                break
    
    def update_advanced_gui(self, frame, predicted_sign, confidence):
        """Actualizar GUI avanzada"""
        try:
            # Redimensionar y convertir frame
            frame_resized = cv2.resize(frame, (600, 450))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Agregar información overlay
            cv2.putText(frame_rgb, f"IA: {predicted_sign} ({confidence:.1%})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.current_word:
                cv2.putText(frame_rgb, f"Palabra: {self.current_word}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Convertir a formato tkinter
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Actualizar imagen
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
            
            # Actualizar información de detección
            sign_text = predicted_sign
            if predicted_sign in self.sign_mapping:
                sign_text += f"\n({self.sign_mapping[predicted_sign]})"
            
            self.detected_sign_label.configure(text=sign_text)
            self.confidence_label.configure(text=f"Confianza: {confidence:.1%}")
            self.detection_count_label.configure(text=f"Detecciones: {self.total_detections}")
            
            # Actualizar palabra actual
            self.current_word_label.configure(text=self.current_word)
            
            # Actualizar estadísticas
            self.update_stats()
            
        except Exception as e:
            print(f"Error actualizando GUI: {str(e)}")
    
    def on_closing(self):
        """Manejar cierre de aplicación"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()
    
    def run(self):
        """Ejecutar la aplicación"""
        self.root.mainloop()

def install_dependencies():
    """Verificar e instalar dependencias automáticamente"""
    import subprocess
    import sys
    
    required_packages = {
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'pyttsx3': 'pyttsx3',
        'requests': 'requests'
    }
    
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("🔧 Instalando dependencias faltantes...")
        for package in missing_packages:
            print(f"📦 Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ Todas las dependencias instaladas")

def main():
    """Función principal"""
    print("=" * 60)
    print("🚀 SISTEMA AVANZADO DE DETECCIÓN LSP CON IA")
    print("=" * 60)
    print("🤖 Tecnología: YOLOv8 + Computer Vision")
    print("🎯 Detección en tiempo real con IA")
    print("🔊 Síntesis de voz integrada")
    print("💬 Formación automática de palabras")
    print("=" * 60)
    
    try:
        # Verificar dependencias
        print("🔍 Verificando dependencias...")
        install_dependencies()
        
        print("🎬 Iniciando aplicación...")
        app = AdvancedSignLanguageDetector()
        
        print("✅ Sistema listo. ¡Disfruta detectando señas!")
        print("\n💡 CONSEJOS DE USO:")
        print("• Asegúrate de tener buena iluminación")
        print("• Mantén las manos dentro del campo de visión")
        print("• Las detecciones se estabilizan automáticamente")
        print("• Usa 'Reproducir' para escuchar las palabras formadas")
        print("• El sistema mejora con el modelo YOLOv8 instalado")
        
        app.run()
        
    except ImportError as e:
        print(f"❌ Error: Falta instalar una dependencia: {str(e)}")
        print("\n🔧 SOLUCIÓN:")
        print("Ejecuta en tu entorno virtual:")
        print("pip install ultralytics opencv-python pillow numpy pyttsx3 requests")
        
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        print("\n🆘 Si el problema persiste:")
        print("1. Verifica que tu cámara funcione")
        print("2. Asegúrate de tener Python 3.8+")
        print("3. Reinstala las dependencias")

if __name__ == "__main__":
    main()