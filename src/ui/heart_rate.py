import numpy as np
import time

class HeartRateMonitor:
    def __init__(self, buffer_size=150, fps=30):
        """
        buffer_size: Câte cadre păstrăm în memorie (150 cadre @ 30fps = 5 secunde de istoric)
        fps: Cadre pe secundă estimate ale webcam-ului
        """
        self.buffer_size = buffer_size
        self.fps = fps
        self.times = []     # Timpul la care a fost luat cadrul
        self.samples = []   # Valoarea medie a canalului Verde
        self.bpm = 0        # Valoarea calculată a pulsului

    def update(self, face_roi):
        """
        Adaugă un nou cadru la analiză.
        face_roi: Imaginea feței detectate (color BGR)
        """
        # 1. Extragem canalul Verde (Green) - indexul 1 în BGR
        # Sângele absoarbe cel mai bine lumina verde, deci aici se vede pulsul
        g_channel = face_roi[:, :, 1]
        
        # 2. Calculăm media intensității pe toată fața
        g_mean = np.mean(g_channel)
        
        # 3. Adăugăm în buffer
        self.samples.append(g_mean)
        self.times.append(time.time())
        
        # Păstrăm doar ultimele 'buffer_size' valori (fereastră glisantă)
        if len(self.samples) > self.buffer_size:
            self.samples.pop(0)
            self.times.pop(0)

        # 4. Calculăm BPM doar dacă avem destule date (ex: > 100 cadre)
        if len(self.samples) > 100:
            self.calculate_bpm()

        return self.bpm

    def calculate_bpm(self):
        # Convertim lista la numpy array pentru viteză
        signal = np.array(self.samples)
        
        # Normalizare (scădem media pentru a centra semnalul în 0)
        signal = signal - np.mean(signal)
        
        # Aplicăm FFT (Fast Fourier Transform) pentru a găsi frecvențele
        # FFT transformă semnalul din "Timp" în "Frecvență"
        fft_result = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1.0/self.fps)
        
        # Filtrăm frecvențele nerealiste
        # Un puls uman normal e între 45 BPM (0.75 Hz) și 180 BPM (3.0 Hz)
        min_freq = 0.75 
        max_freq = 3.0
        
        # Setăm pe zero tot ce e în afara intervalului uman (zgomot)
        fft_result[(freqs < min_freq) | (freqs > max_freq)] = 0
        
        # Găsim frecvența dominantă (cel mai înalt vârf din grafic)
        idx = np.argmax(np.abs(fft_result))
        dominant_freq = freqs[idx]
        
        # Convertim Hz în BPM (Bătăi pe minut)
        new_bpm = dominant_freq * 60.0
        
        # Mică stabilizare (media ponderată) pentru a nu sări brusc cifrele
        self.bpm = 0.9 * self.bpm + 0.1 * new_bpm

        return self.bpm