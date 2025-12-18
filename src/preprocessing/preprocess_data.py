import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# --- CONFIGURARE ---
BASE_DIR = "data"
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "raw/train") # Sursa FER2013 Train
RAW_TEST_DIR = os.path.join(BASE_DIR, "raw/test")   # Sursa FER2013 Test
GENERATED_DIR = os.path.join(BASE_DIR, "generated") # Pozele tale
PROCESSED_DIR = os.path.join(BASE_DIR, "processed") # Output .npy

IMG_SIZE = 48
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def process_folder(folder_path, category, limit=None):
    """
    Citește imaginile dintr-un folder specific.
    """
    images_temp, labels_temp = [], []
    full_path = os.path.join(folder_path, category)
    
    if not os.path.exists(full_path):
        return [], []

    file_names = os.listdir(full_path)
    random.shuffle(file_names) # Amestecăm ca să luăm poze aleatorii din FER
    
    if limit: 
        file_names = file_names[:int(limit)]
        
    class_num = CATEGORIES.index(category)
    
    for img_name in file_names:
        try:
            img_path = os.path.join(full_path, img_name)
            # Citim direct în Grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize la 48x48
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images_temp.append(img)
                labels_temp.append(class_num)
        except Exception: 
            continue
            
    return images_temp, labels_temp

def prepare_data():
    X_train_val_list = []
    y_train_val_list = []
    
    print("📊 Începem balansarea dinamică (40% Proprii / 60% Publice)...")
    
    # 1. Construim setul PRINCIPAL (pentru Train și Validation)
    for cat in CATEGORIES:
        # A. Încărcăm TOATE datele tale (Generated)
        imgs_gen, lbls_gen = process_folder(GENERATED_DIR, cat, limit=None)
        count_gen = len(imgs_gen)
        
        # B. Calculăm câte poze publice (FER) ne trebuie
        if count_gen > 0:
            # Dacă Generated = 40%, atunci FER = 60%.
            # Raportul 60/40 = 1.5. Deci avem nevoie de 1.5 ori mai multe poze publice.
            limit_fer = int(count_gen * 1.5)
        else:
            # Fallback dacă nu ai poze deloc pentru o clasă (ex: Disgust)
            print(f"⚠️ Atenție: Nu există poze generate pentru '{cat}'. Folosesc 500 poze FER default.")
            limit_fer = 500

        # C. Încărcăm datele publice
        imgs_fer, lbls_fer = process_folder(RAW_TRAIN_DIR, cat, limit=limit_fer)
        
        # D. Combinăm
        X_train_val_list.extend(imgs_gen + imgs_fer)
        y_train_val_list.extend(lbls_gen + lbls_fer)
        
        print(f"   -> Clasa '{cat}': {count_gen} proprii + {len(imgs_fer)} publice (Total: {count_gen + len(imgs_fer)})")

    X_all = np.array(X_train_val_list)
    y_all = np.array(y_train_val_list)
    
    # 2. SPLIT: 85% Train / 15% Validation
    # Folosim stratify pentru a păstra proporțiile claselor
    print(f"\n✂️  Împărțire dataset combinat ({len(X_all)} imagini)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, 
        test_size=0.15, 
        stratify=y_all, 
        random_state=42
    )

    # 3. Construim setul de TEST (Doar date publice - Raw Test)
    # Acesta rămâne pur pentru a evalua corect generalizarea
    print("📥 Creare set de TEST (doar date publice)...")
    X_test_list, y_test_list = [], []
    for cat in CATEGORIES:
        # Luăm câte 150-200 de poze per clasă pentru testare rapidă
        imgs_t, lbls_t = process_folder(RAW_TEST_DIR, cat, limit=200)
        X_test_list.extend(imgs_t)
        y_test_list.extend(lbls_t)
    
    X_test, y_test = np.array(X_test_list), np.array(y_test_list)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_npy(X, y, name):
    print(f"💾 Salvare {name} shape: {X.shape}")
    
    # Normalizare (0-1)
    X = X.astype('float32') / 255.0
    
    # Reshape pentru CNN (N, 48, 48, 1)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    # One-Hot Encoding
    y = to_categorical(y, num_classes=len(CATEGORIES))
    
    if not os.path.exists(PROCESSED_DIR): os.makedirs(PROCESSED_DIR)
    
    np.save(os.path.join(PROCESSED_DIR, f"X_{name}.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, f"y_{name}.npy"), y)

if __name__ == "__main__":
    train, val, test = prepare_data()
    
    save_npy(train[0], train[1], "train")
    save_npy(val[0], val[1], "val")
    save_npy(test[0], test[1], "test")
    
    print("\n✅ Procesare completă! Datele sunt salvate în folderul 'processed'.")