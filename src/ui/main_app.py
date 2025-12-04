import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
from heart_rate import HeartRateMonitor # <--- NOU: Importăm clasa noastră

# --- CONFIGURARE ---
MODEL_PATH = 'models/emotion_model.keras'
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def load_sia_model():
    # ... (Codul vechi de încărcare model rămâne la fel) ...
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except:
        return None

def preprocess_face(face_img):
    # 1. Verificăm forma imaginii pentru a vedea dacă e color
    # OpenCV returnează (H, W) pentru grayscale (len=2) și (H, W, 3) pentru color (len=3)
    
    if len(face_img.shape) == 3:
        # Dacă are 3 dimensiuni, verificăm dacă ultima dimensiune este 3 (canale RGB)
        if face_img.shape[2] == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Redimensionare la 48x48 (cum a fost antrenat modelul)
    face_img = cv2.resize(face_img, (48, 48))
    
    # 3. Normalizare (0-1)
    face_img = face_img.astype('float32') / 255.0
    
    # 4. Expand dimensions pentru Keras: (1, 48, 48, 1)
    # Transformăm (48, 48) -> (1, 48, 48, 1)
    face_img = np.expand_dims(face_img, axis=0) # Batch dimension
    face_img = np.expand_dims(face_img, axis=-1) # Channel dimension
    
    return face_img

def main():
    model = load_sia_model()
    if model is None: return

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    
    # --- NOU: Inițializăm monitorul cardiac ---
    # Presupunem webcam la 30 FPS, buffer de 150 cadre (5 secunde)
    hr_monitor = HeartRateMonitor(buffer_size=150, fps=30)

    print("🎥 Pornire cameră... Stai nemișcat pentru puls!")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            # ROI-uri
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w] # Aceasta merge la HeartRateMonitor

            # 1. EMOȚIE (Rețea Neuronală)
            try:
                processed = preprocess_face(roi_gray)
                pred = model.predict(processed, verbose=0)
                label = EMOTIONS[np.argmax(pred)]
                conf = np.max(pred) * 100
            except Exception as e:  # <--- MODIFICARE AICI
                print(f"Eroare la predicție: {e}") # <--- MODIFICARE AICI: Afișăm eroarea în consolă
                label = "Error"
                conf = 0

            # 2. PULS (Algoritm rPPG) <--- NOU
            bpm = hr_monitor.update(roi_color)

            # 3. DIAGNOSTIC COMBINAT (Logica de "Pigment")
            diagnosis = "Normal"
            if label in ["Fear", "Angry"] and bpm > 90:
                diagnosis = "STRES RIDICAT"
                color_diag = (0, 0, 255) # Roșu
            elif label == "Happy" and bpm < 80:
                diagnosis = "Relaxat"
                color_diag = (0, 255, 0) # Verde
            else:
                color_diag = (255, 255, 0) # Galben

            # --- DESENARE ---
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color_diag, 2)
            
            # Text Emoție
            cv2.putText(display_frame, f"{label} ({int(conf)}%)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
            
            # Text Puls + Diagnostic
            info_text = f"BPM: {int(bpm)} | {diagnosis}"
            cv2.putText(display_frame, info_text, (x, y+h+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 64), 2)

        cv2.imshow('SIA - Emotion & Heart Rate', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()