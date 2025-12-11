import os
import cv2
import numpy as np
import random
from keras.utils import to_categorical

# --- CONFIGURARE ---
# Ajustează căile dacă sunt diferite la tine
BASE_DIR = "data"
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "raw/train")   # Folderul original FER2013
RAW_TEST_DIR = os.path.join(BASE_DIR, "raw/test")     # Folderul original de test
GENERATED_DIR = os.path.join(BASE_DIR, "generated")   # Folderul cu datele tale augmentate
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")   # Unde salvăm array-urile

IMG_SIZE = 48
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# CALCUL MATEMATIC:
# Vrem ~4000 imagini din FER. Sunt 7 clase.
# 4000 / 7 = 571.4 -> Rotunjim la 572 imagini per clasă.
LIMIT_FER_PER_CLASS = 572 

def process_folder(folder_path, category, limit=None):
    """
    Citește imaginile dintr-un folder, le face resize și le pune într-o listă.
    """
    images_temp = []
    labels_temp = []
    
    full_path = os.path.join(folder_path, category)
    
    # Verificăm dacă folderul există (ca să nu primim eroare dacă lipsește vreo emoție)
    if not os.path.exists(full_path):
        print(f"⚠️ Atenție: Nu am găsit folderul: {full_path}")
        return [], []

    # Luăm toate fișierele și le amestecăm
    file_names = os.listdir(full_path)
    random.shuffle(file_names)
    
    # Dacă avem limită (ex: doar 572 poze), tăiem lista
    if limit is not None:
        file_names = file_names[:limit]
        
    print(f"   -> Procesez {len(file_names)} imagini din: {folder_path.split('/')[-1]}/{category}")

    class_num = CATEGORIES.index(category) # Ex: "happy" -> 3

    for img_name in file_names:
        try:
            img_path = os.path.join(full_path, img_name)
            
            # 1. Citim imaginea direct în Alb-Negru (Grayscale)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img_array is None:
                continue # Sărim peste fișiere corupte

            # 2. Redimensionăm la 48x48 (standardul CNN-ului nostru)
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            
            images_temp.append(resized_array)
            labels_temp.append(class_num)
            
        except Exception as e:
            pass # Ignorăm erorile punctuale
            
    return images_temp, labels_temp

def create_training_array():
    # Aici construim array-ul final combinat
    X_list = []
    y_list = []
    
    print("🔄 --- PASUL 1: Adăugăm imaginile din FER2013 (Subset 4000) ---")
    for category in CATEGORIES:
        imgs, lbls = process_folder(RAW_TRAIN_DIR, category, limit=LIMIT_FER_PER_CLASS)
        X_list.extend(imgs)
        y_list.extend(lbls)

    print("\n🔄 --- PASUL 2: Adăugăm imaginile TALE (Generate/Augmentate) ---")
    # Aici NU punem limită, luăm tot ce ai generat tu
    if os.path.exists(GENERATED_DIR):
        for category in CATEGORIES:
            # limit=None înseamnă "ia tot"
            imgs, lbls = process_folder(GENERATED_DIR, category, limit=None)
            X_list.extend(imgs)
            y_list.extend(lbls)
    else:
        print("❌ EROARE: Nu găsesc folderul 'data/generated'. Rulează augmentarea mai întâi!")

    # Transformăm listele în NumPy Arrays
    return np.array(X_list), np.array(y_list)

def create_test_array():
    # Pentru testare folosim doar datele originale FER (ca să fim corecți)
    X_list = []
    y_list = []
    
    print("\n🔄 --- PASUL 3: Creăm setul de testare (FER2013) ---")
    for category in CATEGORIES:
        # Luăm doar 100 de imagini de test per clasă pentru viteză
        imgs, lbls = process_folder(RAW_TEST_DIR, category, limit=100) 
        X_list.extend(imgs)
        y_list.extend(lbls)
        
    return np.array(X_list), np.array(y_list)

def finalize_and_save(X, y, name):
    print(f"\n⚙️ Finalizare array {name}...")
    
    # 1. Normalizare: Împărțim la 255 ca să avem valori între 0 și 1
    # Astfel rețeaua învață mai repede
    X = X.astype('float32') / 255.0
    
    # 2. Reshape: Rețeaua vrea formatul (Nr_Poze, 48, 48, 1)
    # Acel '1' de la final înseamnă 1 canal de culoare (Gri)
    if len(X) > 0:
        X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    # 3. One-Hot Encoding la etichete: 3 devine [0,0,0,1,0,0,0]
    y = to_categorical(y, num_classes=len(CATEGORIES))
    
    # 4. Salvare pe disc
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    np.save(os.path.join(PROCESSED_DIR, f"X_{name}.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, f"y_{name}.npy"), y)
    
    print(f"✅ Array salvat: X_{name}.npy (Shape: {X.shape})")

if __name__ == "__main__":
    # Executăm funcțiile
    X_train, y_train = create_training_array()
    finalize_and_save(X_train, y_train, "train")
    
    X_test, y_test = create_test_array()
    finalize_and_save(X_test, y_test, "test")

    print("\n🎉 Gata! Toate imaginile sunt acum în array-uri .npy și gata de antrenare.")