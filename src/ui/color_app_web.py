import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from scipy import signal
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# --- CONFIGURATION ---
MODEL_PATH = 'models/emotion_model_epoch_50.pt'
REPORTS_DIR = "docs/reports"
IMAGE_SIZE = 100
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SETTINGS
FRAME_WINDOW = 6  # Smoothing for emotions
APPLY_GAMMA = True
GAMMA_VALUE = 1.0  # 1.0 is neutral. Lower (0.8) darker, Higher (1.3) brighter.

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
EMOTION_COLORS = {
    "angry": (0, 0, 255),      # Red
    "disgust": (0, 140, 255),  # Orange
    "fear": (255, 0, 255),     # Magenta
    "happy": (0, 255, 0),      # Green
    "neutral": (200, 200, 200),  # Grey
    "sad": (255, 0, 0),        # Blue
    "surprise": (0, 255, 255)  # Yellow
}

# Gamification / Zone Colors (For Face Box)
COLOR_RELAXED = (0, 255, 0)    # Green
COLOR_STRESSED = (0, 0, 255)   # Red
COLOR_NEUTRAL = (255, 255, 0)  # Cyan/Teal


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
        if len(idx[0]) == 0:
            return

        pruned = fft[idx]
        pfreq = freqs[idx]
        self.bpm = pfreq[np.argmax(pruned)]


class SimpleEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleEmotionCNN, self).__init__()

        # Block 1: 3 -> 32 channels. Output size: 50x50
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 32 -> 64 channels. Output size: 25x25
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Block 3: 64 -> 128 channels. Output size: 12x12
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Classifier
        self.fc = nn.Linear(128 * 12 * 12, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out


# --- 2. MODEL UTILS ---
@st.cache_resource
def load_model():
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    model = SimpleEmotionCNN(num_classes=7)
    
    # Load Trained Weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Could not find {MODEL_PATH}")
        return None
    
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
        c_mpl = (c[2]/255, c[1]/255, c[0]/255)  # BGR to RGB
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
    return filename


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


# --- 5. VIDEO PROCESSOR ---
class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = load_model()
        self.hr_monitor = HeartRateMonitor()
        self.prob_buffer = deque(maxlen=FRAME_WINDOW)
        self.current_emotion = "neutral"
        self.current_confidence = 0
        self.current_bpm = 0
        self.current_zone = "Neutral"
        self.is_recording = False
        self.session_data = {'times': [], 'bpm': [], 'probs': []}
        self.apply_gamma = APPLY_GAMMA
        self.gamma_value = GAMMA_VALUE
        
        self.face_lost_counter = 0
        self.face_lost_threshold = 30

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if self.apply_gamma:
            img = adjust_gamma(img, gamma=self.gamma_value)
        
        display_frame = img.copy()
        h, w, _ = display_frame.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        
        current_probs = np.zeros(len(CLASSES))

        if len(faces) > 0 and self.model is not None:
            # Reset face lost counter
            self.face_lost_counter = 0
            
            (x, y, w_face, h_face) = faces[0]
            
            # 1. Heart Rate
            bpm = self.hr_monitor.update(img[y:y+h_face, x:x+w_face])
            self.current_bpm = bpm

            # 2. Emotion Inference
            try:
                face_roi = img[y:y+h_face, x:x+w_face]
                rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    out = self.model(tensor)
                    probs = torch.nn.functional.softmax(out[0], dim=0).cpu().numpy()
                    probs = probs / np.sum(probs)
                    self.prob_buffer.append(probs)
                    
                if len(self.prob_buffer) > 0:
                    current_probs = np.mean(self.prob_buffer, axis=0)

                winner_idx = np.argmax(current_probs)
                label = CLASSES[winner_idx]
                confidence = current_probs[winner_idx]
                
                self.current_emotion = label
                self.current_confidence = confidence

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
                
                self.current_zone = status_text

                # --- 4. DATA LOGGING ---
                if self.is_recording:
                    self.session_data['times'].append(time.time())
                    self.session_data['bpm'].append(bpm)
                    self.session_data['probs'].append(current_probs)

                # --- 5. DRAW UI ---
                # Face Box (Gamified Color)
                cv2.rectangle(display_frame, (x, y), (x+w_face, y+h_face), ui_color, 2)
                
                # Header Background (Gamified Color)
                cv2.rectangle(display_frame, (x, y-40), (x+w_face, y), ui_color, -1)
                
                # Main Label (White Text)
                cv2.putText(display_frame, f"{label.upper()} {int(confidence*100)}%", 
                           (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Heart Rate & Zone Status (Bottom)
                hr_color = (0, 0, 255) if bpm > 100 else (0, 255, 0)
                cv2.putText(display_frame, f"HR: {int(bpm)} BPM", (x, y+h_face+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)
                
                cv2.putText(display_frame, status_text, (x, y+h_face+60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)

                if self.is_recording:
                    cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(display_frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except Exception as e:
                # IMPROVED: Log errors instead of silently passing
                print(f"⚠️ Processing Error: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Show error on frame
                cv2.putText(display_frame, "Processing Error - Check Console", 
                           (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        else:
            # NEW: Handle no face detected
            self.face_lost_counter += 1
            
            # Show warning messages
            warning_y = h // 2 - 50
            
            if self.face_lost_counter < self.face_lost_threshold:
                # Temporary face loss
                cv2.putText(display_frame, "NO FACE DETECTED", 
                           (w//2 - 200, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                cv2.putText(display_frame, "Position your face in the frame", 
                           (w//2 - 220, warning_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Prolonged face loss - clear buffer
                if len(self.prob_buffer) > 0:
                    self.prob_buffer.clear()
                    print("⚠️ Face lost for >1 second - clearing emotion buffer")
                
                cv2.putText(display_frame, "FACE LOST", 
                           (w//2 - 120, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(display_frame, "Move back into frame", 
                           (w//2 - 180, warning_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Reset metrics
                self.current_emotion = "No face"
                self.current_confidence = 0
                self.current_zone = "N/A"
            
            # Pause recording notification
            if self.is_recording:
                cv2.putText(display_frame, "⚠️ RECORDING PAUSED - No Face", 
                           (w//2 - 250, warning_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        
        # --- 6. SIDEBAR STATS (Always visible) ---
        if len(self.prob_buffer) > 0:
            current_probs = np.mean(self.prob_buffer, axis=0)
        draw_bar_chart(display_frame, current_probs, CLASSES)

        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")



# --- 6. STREAMLIT APP ---
def main():
    st.set_page_config(
        page_title="Biometric Monitor",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Biometric Emotion Monitor")
    st.markdown("Real-time emotion detection and heart rate monitoring")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Model status
        model = load_model()
        if model is not None:
            st.success("✅ Model loaded successfully")
        else:
            st.error("❌ Model not found. Please check MODEL_PATH")
            st.stop()
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        gamma_val = st.slider("Gamma Correction", 0.5, 2.0, GAMMA_VALUE, 0.1)
        apply_gamma = st.checkbox("Apply Gamma", value=APPLY_GAMMA)
        
        st.markdown("---")
        
        # Instructions
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Allow camera access when prompted
        2. Position your face in the frame
        3. Use the controls below to start/stop recording
        4. Download your report when done
        """)
        
        st.markdown("---")
        st.info("**Zone Status:**\n- 🟢 Relaxed: Happy + Low HR\n- 🔴 Stressed: Fear/Angry or High HR\n- 🔵 Neutral: Default state")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create video processor context
        ctx = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Update gamma settings in real-time
        if ctx.video_processor:
            ctx.video_processor.apply_gamma = apply_gamma
            ctx.video_processor.gamma_value = gamma_val
    
    with col2:
        st.subheader("📊 Live Metrics")
        
        # Placeholders for live metrics
        emotion_placeholder = st.empty()
        confidence_placeholder = st.empty()
        bpm_placeholder = st.empty()
        zone_placeholder = st.empty()
        
        # Recording controls
        st.markdown("---")
        st.subheader("🎬 Recording")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            if st.button("▶️ Start Recording", use_container_width=True):
                if ctx.video_processor:
                    ctx.video_processor.is_recording = True
                    ctx.video_processor.session_data = {'times': [], 'bpm': [], 'probs': []}
                    st.success("Recording started!")
        
        with col_rec2:
            if st.button("⏹️ Stop & Generate Report", use_container_width=True):
                if ctx.video_processor:
                    was_recording = ctx.video_processor.is_recording
                    ctx.video_processor.is_recording = False
                    
                    # NEW: Validate data before generating report
                    data_points = len(ctx.video_processor.session_data['times'])
                    
                    if was_recording and data_points >= 10:
                        report_path = generate_report(ctx.video_processor.session_data)
                        st.success(f"✅ Report saved! ({data_points} data points)")
                        
                        # Display the report
                        st.image(report_path, caption="Session Report")
                        
                        # Download button
                        with open(report_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Report",
                                data=file,
                                file_name=os.path.basename(report_path),
                                mime="image/png",
                                use_container_width=True
                            )
                    elif was_recording and data_points > 0:
                        st.warning(f"⚠️ Recording stopped but insufficient data ({data_points} frames). Need at least 10 frames for a valid report.")
                        st.info("💡 Try recording for at least 5-10 seconds with your face visible.")
                    else:
                        st.warning("No data recorded yet!")

        # Update live metrics
        if ctx.video_processor:
            emotion_placeholder.metric("Current Emotion", 
                                      ctx.video_processor.current_emotion.upper(), 
                                      f"{int(ctx.video_processor.current_confidence*100)}%")
            bpm_placeholder.metric("Heart Rate", f"{int(ctx.video_processor.current_bpm)} BPM")
            zone_placeholder.metric("Status Zone", ctx.video_processor.current_zone)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>Biometric Monitor v2.0 | Powered by PyTorch & Streamlit</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()