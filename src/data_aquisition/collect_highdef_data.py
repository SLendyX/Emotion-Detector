import cv2
import os
import time

# --- CONFIG ---
OUTPUT_DIR = 'data/generated'
IMAGE_SIZE = 224 # Matches your ResNet training size exactly
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Create folders if they don't exist
for e in EMOTIONS:
    os.makedirs(os.path.join(OUTPUT_DIR, e), exist_ok=True)

cap = cv2.VideoCapture(0)

print("--------------------------------------------------")
print("   DATA COLLECTOR - SAVE FULL COLOR 224x224    ")
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

counts = {e: len(os.listdir(os.path.join(OUTPUT_DIR, e))) for e in EMOTIONS}

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Mirror
    
    # Show the capture zone (center square)
    h, w, _ = frame.shape
    center_y, center_x = h // 2, w // 2
    
    # Calculate crop box (we want a square)
    box_size = 400 # Capture a large square from webcam
    y1 = max(0, center_y - box_size // 2)
    y2 = min(h, center_y + box_size // 2)
    x1 = max(0, center_x - box_size // 2)
    x2 = min(w, center_x + box_size // 2)
    
    # Draw rectangle on screen so you know where to put your face
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add stats to screen
    y_txt = 30
    for e in EMOTIONS:
        cv2.putText(frame, f"{e.upper()}: {counts[e]}", (10, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_txt += 20

    cv2.imshow("Data Collector", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    save_emotion = None
    if key == ord('a'): save_emotion = 'angry'
    elif key == ord('d'): save_emotion = 'disgust'
    elif key == ord('f'): save_emotion = 'fear'
    elif key == ord('h'): save_emotion = 'happy'
    elif key == ord('n'): save_emotion = 'neutral'
    elif key == ord('s'): save_emotion = 'sad'
    elif key == ord('u'): save_emotion = 'surprise'
    elif key == ord('q'): break
    
    if save_emotion:
        # 1. Crop the square
        face_img = frame[y1:y2, x1:x2]
        
        # 2. Resize to 224x224 (The exact input for ResNet)
        # We do this NOW so the saved file is high quality, not upscaled later
        face_img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # 3. Generate filename
        timestamp = int(time.time() * 1000)
        filename = f"{OUTPUT_DIR}/{save_emotion}/img_{timestamp}.jpg"
        
        # 4. Save
        cv2.imwrite(filename, face_img)
        print(f"Saved to {filename}")
        counts[save_emotion] += 1

cap.release()
cv2.destroyAllWindows()