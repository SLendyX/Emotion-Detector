import os
# Ascundem GPU-ul pentru a evita erorile pe sisteme fără NVIDIA configurat
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from datetime import datetime
from heart_rate import HeartRateMonitor # Asigură-te că ai fișierul heart_rate.py creat anterior

# --- CONFIGURARE ---
MODEL_PATH = 'models/best_model.keras'
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
REPORTS_DIR = "docs/reports"

# --- GAMIFICATION: Culori pentru stări ---
COLOR_RELAXED = (0, 255, 0)       # Verde
COLOR_STRESSED = (0, 0, 255)      # Roșu
COLOR_NEUTRAL = (255, 255, 0)     # Turcoaz/Galben
COLOR_RECORDING = (0, 0, 128)     # Roșu închis (pentru butonul REC)

def load_sia_model():
    print(f"🔄 Încărcare model ANTRENAT din {MODEL_PATH}...")
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Eroare la încărcarea modelului .h5: {e}")
        return None

def preprocess_face(face_img):
    # Verificare sigură canale
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

def generate_session_report(history):
    """
    Generează un grafic cu evoluția sesiunii și îl salvează ca imagine.
    """
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    if len(history['times']) < 2:
        print("⚠️ Sesiune prea scurtă pentru raport.")
        return

    # Calculăm durata relativă (secunde de la început)
    start_time = history['times'][0]
    relative_times = [t - start_time for t in history['times']]
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Pulsul
    plt.subplot(2, 1, 1)
    plt.plot(relative_times, history['bpm'], color='red', linewidth=2, label='Heart Rate (BPM)')
    plt.ylabel('BPM')
    plt.title(f'Raport Sesiune - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: Emoțiile
    plt.subplot(2, 1, 2)
    # Convertim emoțiile text în numere pentru plotare simplă
    emo_indices = [EMOTIONS.index(e) if e in EMOTIONS else -1 for e in history['emotions']]
    plt.scatter(relative_times, emo_indices, c=emo_indices, cmap='viridis', s=50, alpha=0.7)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel('Timp (secunde)')
    plt.ylabel('Emoție Detectată')
    plt.grid(True, alpha=0.3)
    
    # Salvare
    filename = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(REPORTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Raport generat: {save_path}")
    return save_path

def main():
    model = load_sia_model()
    if model is None: return

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    hr_monitor = HeartRateMonitor(buffer_size=150, fps=30)

    # Variabile Sesiune
    is_recording = False
    session_data = {'times': [], 'bpm': [], 'emotions': []}
    last_report_path = ""

    print("🎥 Pornire SIA... Comenzi:")
    print("   [R] - Start/Stop Înregistrare Sesiune (Raport)")
    print("   [Q] - Ieșire")

    while True:
        ret, frame = cap.read()
        if not ret: break

        display_frame = frame.copy()
        
        # --- UI: Status Bar ---
        if is_recording:
            cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1) # Bulină roșie REC
            cv2.putText(display_frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Press 'R' to Record", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # 1. Inferență Emoție
            try:
                processed = preprocess_face(roi_gray)
                pred = model.predict(processed, verbose=0)
                emotion_idx = np.argmax(pred)
                label = EMOTIONS[emotion_idx]
                conf = np.max(pred) * 100
            except:
                label = "N/A"
                conf = 0

            # 2. Calcul Puls
            bpm = hr_monitor.update(roi_color)

            # 3. GAMIFICATION & DIAGNOSTIC
            # Schimbăm culoarea chenarului în funcție de starea utilizatorului
            ui_color = COLOR_NEUTRAL
            status_text = "Zona: Neutru"

            if label in ["Fear", "Angry"] or bpm > 100:
                ui_color = COLOR_STRESSED
                status_text = "Zona: STRES / ALERTA"
            elif label == "Happy" and bpm < 85:
                ui_color = COLOR_RELAXED
                status_text = "Zona: Relaxare / Focus"
            
            # --- ÎNREGISTRARE DATE ---
            if is_recording:
                session_data['times'].append(time.time())
                session_data['bpm'].append(bpm)
                session_data['emotions'].append(label)

            # --- DESENARE UI ---
            # Chenar cu colțuri rotunjite (simulat prin linii groase)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), ui_color, 3)
            
            # Header Emoție (Sus)
            cv2.putText(display_frame, f"{label} {int(conf)}%", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_color, 2)
            
            # Footer Date (Jos)
            cv2.putText(display_frame, f"HR: {int(bpm)} bpm", (x, y+h+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Footer Gamification
            cv2.putText(display_frame, status_text, (x, y+h+50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 1)

        # Afișare mesaj dacă tocmai s-a salvat un raport
        if last_report_path:
            cv2.putText(display_frame, "Raport salvat!", (display_frame.shape[1]-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('SIA - Smart Emotion Monitor', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Toggle Înregistrare
            if not is_recording:
                is_recording = True
                session_data = {'times': [], 'bpm': [], 'emotions': []} # Resetăm datele
                last_report_path = ""
                print("⏺️  Sesiune pornită...")
            else:
                is_recording = False
                print("⏹️  Sesiune oprită. Generez raport...")
                last_report_path = generate_session_report(session_data)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()