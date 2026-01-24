import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from scipy import signal
import os

# --- CONFIGURATION ---
MODEL_PATH = 'best_emotion_model.pth'
REPORTS_DIR = "reports"
IMAGE_SIZE = 224
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SETTINGS
FRAME_WINDOW = 6  # Smoothing for emotions
APPLY_GAMMA = True
GAMMA_VALUE = 1.0 # 1.0 is neutral. Lower (0.8) darker, Higher (1.3) brighter.

# FINAL SENSITIVITY TUNING
SENSITIVITY = {
    'angry': 1.2,
    'disgust': 0.8,
    'fear': 3.0,
    'happy': 1.0,
    'neutral': 0.6,
    'sad': 2.5,
    'surprise': 0.9
}

# --- COLORS (BGR for OpenCV) ---
# Specific Emotion Colors for Text/Bars
EMOTION_COLORS = {
    "angry":    (0, 0, 255),      # Red
    "disgust":  (0, 140, 255),    # Orange
    "fear":     (255, 0, 255),    # Magenta
    "happy":    (0, 255, 0),      # Green
    "neutral":  (200, 200, 200),  # Grey
    "sad":      (255, 0, 0),      # Blue
    "surprise": (0, 255, 255)     # Yellow
}

# Gamification / Zone Colors (For Face Box)
COLOR_RELAXED = (0, 255, 0)       # Green
COLOR_STRESSED = (0, 0, 255)      # Red
COLOR_NEUTRAL = (255, 255, 0)     # Cyan/Teal

# --- 1. HEART RATE MONITOR (rPPG) ---
class HeartRateMonitor:
    def __init__(self, buffer_size=150, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.times = deque(maxlen=buffer_size)
        self.greens = deque(maxlen=buffer_size)
        self.bpm = 0
        self.last_update = time.time()

    def update(self, face_roi):
        """Calculates Heart Rate by analyzing average Green intensity."""
        g = np.mean(face_roi[:, :, 1])
        self.greens.append(g)
        self.times.append(time.time())

        if len(self.greens) > self.buffer_size // 2 and (time.time() - self.last_update) > 0.5:
            self.last_update = time.time()
            self.calculate_bpm()
            
        return self.bpm

    def calculate_bpm(self):
        y = np.array(self.greens)
        y = signal.detrend(y)
        L = len(y)
        even_times = np.linspace(self.times[0], self.times[-1], L)
        y_interp = np.interp(even_times, self.times, y)
        y_interp = y_interp * np.hamming(L)
        raw = np.fft.rfft(y_interp)
        fft = np.abs(raw)
        freqs = float(self.fps) / L * np.arange(L / 2 + 1) * 60.
        
        idx = np.where((freqs > 45) & (freqs < 180))
        if len(idx[0]) == 0: return

        pruned = fft[idx]
        pfreq = freqs[idx]
        self.bpm = pfreq[np.argmax(pruned)]

# --- 2. MODEL UTILS ---
def load_model():
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
    model = model.to(DEVICE)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- 3. REPORT GENERATOR ---
def generate_report(session_data):
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    times = np.array(session_data['times']) - session_data['times'][0]
    bpm = session_data['bpm']
    probs = np.array(session_data['probs'])

    plt.figure(figsize=(12, 10))

    # Plot 1: Heart Rate
    plt.subplot(2, 1, 1)
    plt.plot(times, bpm, color='red', label='Heart Rate (BPM)', linewidth=2)
    plt.title('Physiological Response')
    plt.ylabel('BPM')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Emotions
    plt.subplot(2, 1, 2)
    for i, label in enumerate(CLASSES):
        c = EMOTION_COLORS[label]
        c_mpl = (c[2]/255, c[1]/255, c[0]/255) # BGR to RGB
        if np.mean(probs[:, i]) > 0.05:
            plt.plot(times, probs[:, i] * 100, label=label.upper(), color=c_mpl, linewidth=2)

    plt.title('Emotional Response')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 105)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{REPORTS_DIR}/Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"✅ Report saved to {filename}")

# --- 4. UI DRAWING UTILS ---
def draw_bar_chart(frame, probs, classes):
    h, w, _ = frame.shape
    bar_width = 150
    start_x = 10
    start_y = 100
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    winner_idx = np.argmax(probs)

    for i, (prob, label) in enumerate(zip(probs, classes)):
        y = start_y + (i * 35)
        color = (255, 255, 255)
        
        if i == winner_idx:
            color = (0, 255, 0)
        
        if label in ['fear', 'sad'] and prob > 0.20 and i != winner_idx:
            color = (0, 255, 255) 

        cv2.putText(frame, f"{label.upper()}", (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(frame, (start_x + 80, y - 10), (start_x + 80 + bar_width, y + 5), (50, 50, 50), -1)
        fill_width = int(prob * bar_width)
        cv2.rectangle(frame, (start_x + 80, y - 10), (start_x + 80 + fill_width, y + 5), color, -1)
        cv2.putText(frame, f"{int(prob*100)}%", (start_x + 90 + bar_width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# --- 5. MAIN APPLICATION ---
def main():
    model = load_model()
    hr_monitor = HeartRateMonitor()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # Session State
    is_recording = False
    session_data = {'times': [], 'bpm': [], 'probs': []}
    prob_buffer = deque(maxlen=FRAME_WINDOW)

    print("\n--- BIOMETRIC MONITOR STARTED ---")
    print("[R] Start/Stop Recording Report")
    print("[Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        if APPLY_GAMMA:
            frame = adjust_gamma(frame, gamma=GAMMA_VALUE)
        
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        
        current_probs = np.zeros(len(CLASSES))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # 1. Heart Rate
            bpm = hr_monitor.update(frame[y:y+h, x:x+w])

            # 2. Emotion Inference
            try:
                face_roi = frame[y:y+h, x:x+w]
                rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    out = model(tensor)
                    probs = torch.nn.functional.softmax(out[0], dim=0).cpu().numpy()
                    for i, cls in enumerate(CLASSES):
                        probs[i] *= SENSITIVITY.get(cls, 1.0)
                    probs = probs / np.sum(probs)
                    prob_buffer.append(probs)
                    
                if len(prob_buffer) > 0:
                    current_probs = np.mean(prob_buffer, axis=0)

                winner_idx = np.argmax(current_probs)
                label = CLASSES[winner_idx]
                confidence = current_probs[winner_idx]

                # --- 3. GAMIFICATION LOGIC (Face Box Color) ---
                ui_color = COLOR_NEUTRAL
                status_text = "Zone: Neutral"
                
                # Condition: High Stress (Fear/Angry OR High Heart Rate)
                if label in ['fear', 'angry'] or bpm > 100:
                    ui_color = COLOR_STRESSED
                    status_text = "Zone: STRESS"
                # Condition: Relaxed (Happy AND Low Heart Rate)
                elif label == 'happy' and bpm < 85:
                    ui_color = COLOR_RELAXED
                    status_text = "Zone: Relaxed"

                # --- 4. DATA LOGGING ---
                if is_recording:
                    session_data['times'].append(time.time())
                    session_data['bpm'].append(bpm)
                    session_data['probs'].append(current_probs)

                # --- 5. DRAW UI ---
                # Face Box (Gamified Color)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), ui_color, 2)
                
                # Header Background (Gamified Color)
                cv2.rectangle(display_frame, (x, y-40), (x+w, y), ui_color, -1)
                
                # Main Label (White Text)
                cv2.putText(display_frame, f"{label.upper()} {int(confidence*100)}%", 
                           (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                # Heart Rate & Zone Status (Bottom)
                hr_color = (0, 0, 255) if bpm > 100 else (0, 255, 0)
                cv2.putText(display_frame, f"HR: {int(bpm)} BPM", (x, y+h+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)
                
                cv2.putText(display_frame, status_text, (x, y+h+60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)

                if is_recording:
                    cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(display_frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as e:
                pass
        
        # --- 6. SIDEBAR STATS (Always visible) ---
        if len(prob_buffer) > 0:
            current_probs = np.mean(prob_buffer, axis=0)
        draw_bar_chart(display_frame, current_probs, CLASSES)

        cv2.imshow("Biometric Monitor", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not is_recording:
                print("Recording Started...")
                session_data = {'times': [], 'bpm': [], 'probs': []}
                is_recording = True
            else:
                print("Recording Stopped.")
                is_recording = False
                generate_report(session_data)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()