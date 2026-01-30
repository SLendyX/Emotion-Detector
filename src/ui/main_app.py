import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from datetime import datetime
from heart_rate import HeartRateMonitor

# --- CONFIGURARE ---
MODEL_PATH = 'models/optimized_model.h5'
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
REPORTS_DIR = "docs/reports"

# --- GAMIFICATION: Culori Emoții (Format BGR pentru OpenCV) ---
# Aceste culori vor fi folosite atât Live cât și în Raport
EMOTION_COLORS = {
    "Angry":    (0, 0, 255),      # Roșu
    "Disgust":  (0, 140, 255),    # Portocaliu
    "Fear":     (255, 0, 255),    # Magenta
    "Happy":    (0, 255, 0),      # Verde
    "Neutral":  (200, 200, 200),  # Gri deschis (pentru vizibilitate pe negru)
    "Sad":      (255, 0, 0),      # Albastru
    "Surprise": (0, 255, 255)     # Galben
}

# Culori Stări (Chenar)
COLOR_RELAXED = (0, 255, 0)       
COLOR_STRESSED = (0, 0, 255)      
COLOR_NEUTRAL = (255, 255, 0)     

def bgr_to_mpl(bgr):
    """Conversie BGR (0-255) la RGB (0-1) pentru Matplotlib"""
    return (bgr[2]/255, bgr[1]/255, bgr[0]/255)

def load_sia_model():
    print(f"🔄 Încărcare model ANTRENAT din {MODEL_PATH}...")
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Eroare la încărcarea modelului: {e}")
        return None

def preprocess_face(face_img):
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

def generate_session_report(history):
    """
    Generează raportul vizual.
    history['probs'] este o listă de array-uri cu toate probabilitățile (7 emoții) per frame.
    """
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    if len(history['times']) < 2:
        print("⚠️ Sesiune prea scurtă pentru raport.")
        return ""

    start_time = history['times'][0]
    relative_times = [t - start_time for t in history['times']]
    
    # Convertim lista de array-uri într-un numpy array mare (Frames x 7)
    # Asta ne permite să extragem coloanele pentru fiecare emoție ușor
    prob_matrix = np.array(history['probs']) 

    plt.figure(figsize=(12, 10))
    
    # --- PLOT 1: PULS (HR) ---
    plt.subplot(2, 1, 1)
    plt.plot(relative_times, history['bpm'], color='red', linewidth=2, label='Heart Rate')
    plt.ylabel('BPM')
    plt.title(f'Raport Sesiune - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # --- PLOT 2: EVOLUȚIE DETALIATĂ EMOȚII (Top 3 vizibil prin linii) ---
    plt.subplot(2, 1, 2)
    
    # Desenăm o linie pentru fiecare emoție folosind culoarea ei specifică
    for i, emotion in enumerate(EMOTIONS):
        # Extragem seria de timp pentru emoția 'i'
        emotion_series = prob_matrix[:, i] * 100 # Convertim în procent 0-100
        
        # Luăm culoarea din dicționar și o convertim pentru matplotlib
        color_bgr = EMOTION_COLORS.get(emotion, (128, 128, 128))
        color_rgb = bgr_to_mpl(color_bgr)
        
        # Plotăm linia
        plt.plot(relative_times, emotion_series, label=emotion, color=color_rgb, linewidth=2, alpha=0.8)

    plt.ylabel('Probabilitate (%)')
    plt.xlabel('Timp (secunde)')
    plt.title('Dinamica Emoțiilor (Toate emoțiile monitorizate)')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.) # Legenda în afara graficului
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105) # Fixăm axa Y între 0 și 100%
    
    # Salvare
    filename = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(REPORTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Raport generat: {save_path}")
    return save_path

def main():
    # --- STATE: Initialization / IDLE ---
    state = "IDLE"
    print(f"🔄 State: {state}")
    
    model = load_sia_model()
    if model is None: 
        print("❌ CRITICAL: Model not found. Exiting.")
        return

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    hr_monitor = HeartRateMonitor(buffer_size=150, fps=30)

    # Variabile Sesiune
    is_recording = False
    session_data = {'times': [], 'bpm': [], 'probs': []}
    last_report_path = ""
    
    # Variabile Temporare per Frame
    frame = None
    display_frame = None
    faces = []
    
    print("🎥 Pornire SIA... Comenzi: [R] Record | [Q] Quit")
    state = "ACQUISITION" # Start loop

    while True:
        # ---------------------------
        # STATE: ACQUISITION (Video Capture)
        # ---------------------------
        if state == "ACQUISITION":
            ret, frame = cap.read()
            if not ret: 
                break
            display_frame = frame.copy()
            
            # Indicator REC
            if is_recording:
                cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1) 
                cv2.putText(display_frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            state = "DETECTION"

        # ---------------------------
        # STATE: DETECTION (Face Detection)
        # ---------------------------
        elif state == "DETECTION":
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            
            if len(faces) > 0:
                state = "PROCESSING"
            else:
                state = "DISPLAY"

        # ---------------------------
        # STATE: PROCESSING & INFERENCE
        # ---------------------------
        elif state == "PROCESSING":
            for (x, y, w, h) in faces:
                roi_gray = gray_frame[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # 1. Inferență Emoție
                top_3_data = [] 
                current_probs = np.zeros(7)
                main_label = "N/A"
                main_conf = 0
                
                try:
                    processed = preprocess_face(roi_gray)
                    pred = model.predict(processed, verbose=0)[0]
                    current_probs = pred
                    
                    sorted_indices = np.argsort(pred)[::-1]
                    top_3_indices = sorted_indices[:3]
                    
                    main_idx = top_3_indices[0]
                    main_label = EMOTIONS[main_idx]
                    main_conf = pred[main_idx] * 100
                    
                    for idx in top_3_indices:
                        top_3_data.append((EMOTIONS[idx], pred[idx] * 100))
                except Exception as e:
                    print(f"Inference Error: {e}")

                # 2. Puls (Fiziologic)
                bpm = hr_monitor.update(roi_color)

                # 3. Gamification Logic
                ui_color = COLOR_NEUTRAL
                status_text = "Zona: Neutru"
                if main_label in ["Fear", "Angry"] or bpm > 100:
                    ui_color = COLOR_STRESSED
                    status_text = "Zona: STRES"
                elif main_label == "Happy" and bpm < 85:
                    ui_color = COLOR_RELAXED
                    status_text = "Zona: Relaxare"
                
                # --- LOGICA RECORDING (în PROCESSING) ---
                if is_recording:
                    session_data['times'].append(time.time())
                    session_data['bpm'].append(bpm)
                    session_data['probs'].append(current_probs)

                # --- DESENARE PRELIMINARĂ PE FRAME ---
                # Modificăm direct display_frame aici
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), ui_color, 3)
                main_emo_color = EMOTION_COLORS.get(main_label, (255,255,255))
                cv2.putText(display_frame, f"{main_label} {int(main_conf)}%", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_emo_color, 2)
                
                text_x = x + w + 10
                text_y = y + 20
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (text_x - 5, y), (text_x + 180, y + 85), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0, display_frame)

                for i, (ename, escore) in enumerate(top_3_data):
                    scolor = EMOTION_COLORS.get(ename, (200, 200, 200))
                    cv2.putText(display_frame, f"{ename}: {int(escore)}%", 
                                (text_x, text_y + (i * 25)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, scolor, 2)
                
                cv2.putText(display_frame, f"HR: {int(bpm)} bpm", (x, y+h+25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, status_text, (x, y+h+50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 1)

            state = "DISPLAY"

        # ---------------------------
        # STATE: DISPLAY / FEEDBACK
        # ---------------------------
        elif state == "DISPLAY":
            if last_report_path:
                cv2.putText(display_frame, "Raport salvat!", (display_frame.shape[1]-200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow('SIA - Smart Emotion Monitor', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not is_recording:
                    is_recording = True
                    session_data = {'times': [], 'bpm': [], 'probs': []} 
                    last_report_path = ""
                    print("⏺️  Sesiune pornită...")
                else:
                    is_recording = False
                    print("⏹️  Sesiune oprită. Generez raport...")
                    last_report_path = generate_session_report(session_data)
            
            # Bucla se reia
            state = "ACQUISITION"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()