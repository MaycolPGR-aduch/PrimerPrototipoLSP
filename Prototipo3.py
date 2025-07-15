import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk

# Configuración
BUFFER_SIZE = 10  # tamaño del buffer para suavizado

# Cargar modelo preentrenado
model = load_model('sign_language_mnist_cnn.h5')
labels = [chr(i) for i in range(65, 91)]

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Inicializar buffer de predicciones
pred_buffer = deque(maxlen=BUFFER_SIZE)

# Tkinter GUI
root = tk.Tk()
root.title("Traductor LSP Mejorado")

lmain = tk.Label(root)
lmain.pack()

pred_label = tk.Label(root, text="Predicción: ", font=("Helvetica", 24))
pred_label.pack()

cap = cv2.VideoCapture(0)

def predict_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, predict_frame)
        return

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    h, w, _ = frame.shape
    gesture_text = ""

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (28, 28))
                roi_normalized = roi_resized / 255.0
                roi_reshaped = roi_normalized.reshape(1, 28, 28, 1)

                pred = model.predict(roi_reshaped, verbose=0)
                idx = np.argmax(pred)
                gesture_text = labels[idx]

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Suavizar predicción
    pred_buffer.append(gesture_text)
    if pred_buffer:
        stable_pred = max(set(pred_buffer), key=pred_buffer.count)
    else:
        stable_pred = ""

    pred_label.config(text=f"Predicción: {stable_pred}")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, predict_frame)

root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
predict_frame()
root.mainloop()
