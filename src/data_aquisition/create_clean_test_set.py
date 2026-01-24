import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical

#path
CSV_PATH = "ckextended.csv"
OUTPUT_DIR = "data/processed"

# Ordinea proiectului tau:
# ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
PROJECT_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

# Mapping CK+ (bazat pe documentatia standard CK+):
# 0=Anger, 1=Disgust, 2=Fear, 3=Happy, 4=Sadness, 5=Surprise, 6=Neutral
CK_TO_PROJECT_MAP = {
    0: 0, # Anger -> Angry
    1: 1, # Disgust -> Disgust
    2: 2, # Fear -> Fear
    3: 3, # Happy -> Happy
    4: 5, # Sadness -> Sadness (idx 5 in project)
    5: 6, # Surprise -> Surprise (idx 6 in project)
    6: 4  # Neutral -> Neutral (idx 4 in project)
}

def create_clean_test_set():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Nu gasesc {CSV_PATH}. Asigura-te ca e in root.")
        return

    print(f"📖 Citire {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Detectare coloane
    pixel_col = [c for c in df.columns if 'pixel' in c.lower() or 'image' in c.lower()][0]
    emotion_col = [c for c in df.columns if 'emotion' in c.lower() or 'label' in c.lower()][0]

    images = []
    labels = []

    print("🔄 Procesare si remapare etichete...")
    for idx, row in df.iterrows():
        ck_label = int(row[emotion_col])
        
        # Ignoram etichete care nu exista in maparea noastra (ex: Contempt=7)
        if ck_label in CK_TO_PROJECT_MAP:
            # 1. Remapam label-ul la standardul proiectului
            project_label = CK_TO_PROJECT_MAP[ck_label]
            
            # 2. Procesam imaginea
            pixels = np.fromstring(str(row[pixel_col]), sep=' ')
            
            # Reshape la 48x48
            if len(pixels) == 2304:
                face = pixels.reshape(48, 48)
                face = face.astype('float32') / 255.0 # Normalizare
                images.append(face)
                labels.append(project_label)

    # Convertim la numpy arrays
    X_test_clean = np.array(images).reshape(-1, 48, 48, 1)
    y_test_clean = to_categorical(np.array(labels), num_classes=7)

    print(f"✅ Generat set test curat: {len(X_test_clean)} imagini.")
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    np.save(os.path.join(OUTPUT_DIR, "X_test_clean.npy"), X_test_clean)
    np.save(os.path.join(OUTPUT_DIR, "y_test_clean.npy"), y_test_clean)
    print(f"💾 Salvat in {OUTPUT_DIR}/X_test_clean.npy")

if __name__ == "__main__":
    create_clean_test_set()
