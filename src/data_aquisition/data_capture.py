import cv2
import os
import time

# Unde salvăm datele noi
OUTPUT_DIR = "data/generated"
EMOTION_TO_CAPTURE = "fear" # Schimbă manual când vrei să capturezi altceva

def capture_images():
    save_path = os.path.join(OUTPUT_DIR, EMOTION_TO_CAPTURE)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    print(f"📸 Începem captura pentru emoția: '{EMOTION_TO_CAPTURE}'.")
    print("Apasă 's' pentru a salva o poză, 'q' pentru a ieși.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        display = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Dacă userul apasă 's', salvăm doar fața decupată
            if cv2.waitKey(1) & 0xFF == ord('s'):
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (48, 48))
                
                filename = os.path.join(save_path, f"{EMOTION_TO_CAPTURE}_{int(time.time())}_{count}.jpg")
                cv2.imwrite(filename, face_roi)
                print(f"✅ Salvat: {filename}")
                count += 1

        cv2.imshow("Data Acquisition", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()