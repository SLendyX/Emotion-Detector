import cv2
import numpy as np
import tensorflow as tf
import os

# --- CONFIGURARE ---
MODEL_PATH = 'models/emotion_model.keras' # Asigură-te că calea e corectă
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def load_sia_model():
    print(f"🔄 Încărcare model din {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model încărcat cu succes!")
        return model
    except Exception as e:
        print(f"❌ Eroare critică: Nu pot încărca modelul. Ai rulat train.py? \nDetalii: {e}")
        return None

def preprocess_face(face_img):
    # 1. Convertire la grayscale (dacă nu e deja)
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # 2. Redimensionare la 48x48 (cum a fost antrenat modelul)
    face_img = cv2.resize(face_img, (48, 48))
    
    # 3. Normalizare (0-1)
    face_img = face_img.astype('float32') / 255.0
    
    # 4. Expand dimensions pentru Keras: (1, 48, 48, 1)
    face_img = np.expand_dims(face_img, axis=0) # Batch dimension
    face_img = np.expand_dims(face_img, axis=-1) # Channel dimension
    
    return face_img

def analyze_physiological(face_roi_color):
    """
    Placeholder pentru partea unică a proiectului (analiza pigment/rPPG).
    Momentan returnează o valoare dummy pentru a demonstra funcționarea pipeline-ului.
    """
    # Aici vei implementa logica de culoare mai târziu
    avg_color_per_row = np.average(face_roi_color, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    
    # Simplu exemplu: Verificăm canalul Roșu vs Verde
    # BGR format in OpenCV: [0]=Blue, [1]=Green, [2]=Red
    red_intensity = avg_color[2]
    green_intensity = avg_color[1]
    
    if red_intensity > green_intensity + 20:
        return "High Arousal (Red)"
    else:
        return "Normal"

def main():
    # 1. Inițializare
    model = load_sia_model()
    if model is None: return

    cap = cv2.VideoCapture(0) # 0 pentru webcam
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)

    print("🎥 Pornire cameră... Apasă 'q' pentru a ieși.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Copie pentru afișare
        display_frame = frame.copy()
        
        # Detecție fețe (pe grayscale)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # -- A. Extragere ROI (Region of Interest) --
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # -- B. Procesare Neuronală (Emoție) --
            try:
                processed_face = preprocess_face(roi_gray)
                prediction = model.predict(processed_face, verbose=0)
                emotion_idx = np.argmax(prediction)
                emotion_label = EMOTIONS[emotion_idx]
                confidence = np.max(prediction) * 100
            except Exception as e:
                print(f"Eroare inferență: {e}")
                continue

            # -- C. Procesare Fiziologică (Culoare) --
            physio_status = analyze_physiological(roi_color)

            # -- D. Desenare UI pe frame --
            # Dreptunghi față
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Text Emoție
            text_emo = f"{emotion_label} ({confidence:.1f}%)"
            cv2.putText(display_frame, text_emo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # Text Fiziologic (Sub față)
            cv2.putText(display_frame, f"Physio: {physio_status}", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

            # Bară de progres pentru încredere
            cv2.rectangle(display_frame, (x, y+h+35), (x+int(w*(confidence/100)), y+h+45), (0, 255, 0), -1)

        # Afișare
        cv2.imshow('SIA - Emotion & Physio Detector', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()