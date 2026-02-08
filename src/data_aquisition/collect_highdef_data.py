import cv2
import os
import time

# --- CONFIG ---
OUTPUT_DIR = 'data/generated'
IMAGE_SIZE = 100  # Target size for the AI
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Load the Haar Cascade (standard OpenCV face detector)
# This comes built-in with OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create folders if they don't exist
for e in EMOTIONS:
    os.makedirs(os.path.join(OUTPUT_DIR, e), exist_ok=True)

cap = cv2.VideoCapture(0)

print("--------------------------------------------------")
print("   DATA COLLECTOR - HAAR FACE DETECT (COLOR)      ")
print("--------------------------------------------------")
print("Press these keys to save an image:")
print("  'a': Angry")
print("  'd': Disgust")
print("  'f': Fear")
print("  'h': Happy")
print("  'n': Neutral")
print("  's': Sad")
print("  'u': Surprise")
print("  'q': QUIT")
print("--------------------------------------------------")

counts = {}
for e in EMOTIONS:
    if os.path.exists(os.path.join(OUTPUT_DIR, e)):
        counts[e] = len(os.listdir(os.path.join(OUTPUT_DIR, e)))
    else:
        counts[e] = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Mirror effect
    h_img, w_img, _ = frame.shape

    # 1. Convert to grayscale for detection (Haar needs gray)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Detect faces
    # scaleFactor=1.1, minNeighbors=5 are standard tuning params
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    face_found = False
    face_coords = None

    # If faces are found, pick the largest one (closest to camera)
    if len(faces) > 0:
        face_found = True
        # Logic: find the face with the biggest area (w * h)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # --- OPTIONAL: Add Padding (so the crop isn't too tight) ---
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        
        face_coords = (x1, y1, x2, y2)

        # Draw rectangle on the screen (Green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        # Visual cue that no face is detected (Red text)
        cv2.putText(frame, "NO FACE DETECTED", (w_img//2 - 100, h_img//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display counts
    y_txt = 30
    for e in EMOTIONS:
        cv2.putText(frame, f"{e.upper()}: {counts[e]}", (10, y_txt), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_txt += 20

    cv2.imshow("Data Collector (Haar)", frame)
    
    # Input handling
    key = cv2.waitKey(1) & 0xFF
    
    save_emotion = None
    if key == ord('a'): save_emotion = 'angry'
    elif key == ord('d'): save_emotion = 'disgust'
    elif key == ord('f'): save_emotion = 'fear'
    elif key == ord('h'): save_emotion = 'happy'
    elif key == ord('n'): save_emotion = 'neutral'
    elif key == ord('s'): save_emotion = 'sad'
    elif key == ord('u'): save_emotion = 'surprise' # Note: 'u' for surprise
    elif key == ord('q'): break
    
    if save_emotion:
        if face_found and face_coords:
            x1, y1, x2, y2 = face_coords
            
            # 3. Crop from the COLOR frame
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size > 0:
                # 4. Resize to 100x100
                face_img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
                
                # 5. Save
                timestamp = int(time.time() * 1000)
                filename = f"{OUTPUT_DIR}/{save_emotion}/img_{timestamp}.jpg"
                cv2.imwrite(filename, face_img)
                
                print(f"Saved {save_emotion} to {filename}")
                counts[save_emotion] += 1
            else:
                print("Error: Crop region invalid.")
        else:
            print("⚠️ Cannot save: No face detected in frame!")

cap.release()
cv2.destroyAllWindows()