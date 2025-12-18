import cv2
import os
import numpy as np
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# CONFIGURARE
INPUT_DIR = "data/generated"  # Aici pui pozele tale originale (făcute cu capture_data.py)
AUGMENT_FACTOR = 10           # Din 1 poză facem 10

# Definim generatorul de augmentări
datagen = ImageDataGenerator(
    rotation_range=15,      # Rotește imaginea cu +/- 15 grade
    width_shift_range=0.1,  # Mută stânga-dreapta
    height_shift_range=0.1, # Mută sus-jos
    shear_range=0.1,        # Deformare ușoară
    zoom_range=0.1,         # Zoom in/out
    horizontal_flip=True,   # Oglindire (important!)
    brightness_range=[0.8, 1.2], # Mai întunecat / mai luminos
    fill_mode='nearest'
)

def augment_my_data():
    print("🚀 Începem augmentarea datelor personale...")
    
    # Parcurgem fiecare folder de emoție (neutral, happy, etc.)
    for emotion in os.listdir(INPUT_DIR):
        emotion_path = os.path.join(INPUT_DIR, emotion)
        
        if not os.path.isdir(emotion_path):
            continue
            
        print(f"📂 Procesez folderul: {emotion}")
        
        # Luăm fiecare poză originală
        for fname in os.listdir(emotion_path):
            if "aug" in fname: 
                continue # Sărim peste pozele deja augmentate dacă rulăm scriptul de 2 ori
                
            img_path = os.path.join(emotion_path, fname)
            
            try:
                # Încărcăm imaginea
                img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape) # Reshape la (1, 48, 48, 1)

                # Generăm imaginile noi
                i = 0
                for batch in datagen.flow(x, batch_size=1, 
                                          save_to_dir=emotion_path, 
                                          save_prefix=f"aug_{fname[:-4]}", 
                                          save_format='jpg'):
                    i += 1
                    if i >= AUGMENT_FACTOR:
                        break # Ne oprim după 10 imagini generate
            except Exception as e:
                print(f"Eroare la {fname}: {e}")

    print("✅ Augmentare completă! Verifică folderele.")

if __name__ == "__main__":
    augment_my_data()