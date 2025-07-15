import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Cargar modelo preentrenado
model = load_model('sign_language_mnist_cnn.h5')

# Etiquetas A-Z
labels = [chr(i) for i in range(65, 91)]

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # espejo
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture_text = ""
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Dibujar landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Obtener bounding box de la mano
            h, w, c = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Expandir un poco la caja
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Recortar ROI
            roi = frame[y_min:y_max, x_min:x_max]

            # Preprocesar para modelo
            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (28, 28))
                roi_normalized = roi_resized / 255.0
                roi_reshaped = roi_normalized.reshape(1, 28, 28, 1)

                # Predecir
                pred = model.predict(roi_reshaped, verbose=0)
                idx = np.argmax(pred)
                gesture_text = labels[idx]

            # Dibujar caja y predicción
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, f"{gesture_text}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Prototipo LSP Mejorado", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
