import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.utils import img_to_array

# --- CONFIGURATION ---
BASE_DIR = "data"
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "raw/train") 
RAW_TEST_DIR = os.path.join(BASE_DIR, "raw/test")   
GENERATED_DIR = os.path.join(BASE_DIR, "generated") 
KDEF_DIR = os.path.join(BASE_DIR, "kdef_cropped")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed") 

IMG_SIZE = 100
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

# --- 1. TRAIN AUGMENTATION (Aggressive) ---
# We want massive variety for the model to learn from.
TRAIN_MULTIPLIERS = {
    "angry": 1,      
    "disgust": 1,    
    "fear": 1,      
    "happy": 1,      
    "neutral": 1,
    "sad": 1,
    "surprised": 1
}

MAX_TRAIN_IMAGES = {
    "angry": 4000,      
    "disgust": 4000,    # FER doesn't have this many, so it will just take 100% of them
    "fear": 4000,       
    "happy": 4000,      # <--- HEAVY CUT: FER has 7200, we only take 3000
    "neutral": 4000,    # <--- HEAVY CUT: FER has 4900, we only take 3000
    "sad": 4000,
    "surprised": 4000
}

# --- 2. VALIDATION AUGMENTATION (Conservative) ---
# We augment validation just enough to make the class sizes comparable.
# This prevents "Happy" from dominating the validation accuracy score.

aug_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_offline(images, multiplier):
    """
    Takes a list of raw images and returns a list of augmented versions.
    """
    if not images or multiplier <= 1:
        return images # Return original if no boost needed

    augmented_images = []
    
    for img in images:
        # 1. Add original
        augmented_images.append(img)
        
        # 2. Reshape
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        # 3. Generate variations
        i = 0
        for batch in aug_datagen.flow(x, batch_size=1):
            aug_img = batch[0].astype('uint8')
            aug_img = aug_img.reshape(IMG_SIZE, IMG_SIZE)
            
            augmented_images.append(aug_img)
            i += 1
            if i >= multiplier: # Stop when we reach the multiplier
                break
                
    return augmented_images

def load_images_from_folder(folder_path, category, limit=None):
    images = []
    labels = []
    full_path = os.path.join(folder_path, category)
    
    if not os.path.exists(full_path):
        return [], []

    file_names = os.listdir(full_path)
    random.shuffle(file_names)
    if limit and len(file_names) > limit:
        print(f"   ✂️ Shaving '{category}' from {len(file_names)} down to {limit}...")
        file_names = file_names[:limit]
    
    class_num = CATEGORIES.index(category)
    
    for img_name in file_names:
        try:
            img_path = os.path.join(full_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(class_num)
        except Exception:
            continue
    return images, labels

def prepare_data():
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []
    
    print(f"📊 Processing Data with Balanced Validation...")

    for cat in CATEGORIES:
        # Get multipliers
        train_mult = TRAIN_MULTIPLIERS.get(cat, 10)

        # --- 1. Load USER Data ---
        my_imgs, my_lbls = load_images_from_folder(GENERATED_DIR, cat, MAX_TRAIN_IMAGES[cat])
        
        if len(my_imgs) > 0:
            if len(my_imgs) < 5:
                train_imgs_my, val_imgs_my = my_imgs, []
                train_lbls_my, val_lbls_my = my_lbls, []
            else:
                train_imgs_my, val_imgs_my, train_lbls_my, val_lbls_my = train_test_split(
                    my_imgs, my_lbls, test_size=0.2, random_state=42
                )
            
            # Augment USER Train AND Val
            train_imgs_my = augment_offline(train_imgs_my, 5)
            
            # Re-generate labels
            train_lbls_my = [CATEGORIES.index(cat)] * len(train_imgs_my)
            val_lbls_my = [CATEGORIES.index(cat)] * len(val_imgs_my)
        else:
            train_imgs_my, val_imgs_my = [], []
            train_lbls_my, val_lbls_my = [], []

        # --- 1.5 Load KDEF Data ---
        kdef_imgs, kdef_lbls = load_images_from_folder(KDEF_DIR, cat)
        
        if len(kdef_imgs) > 0:
            if len(kdef_imgs) < 5:
                train_imgs_kdef, val_imgs_kdef = kdef_imgs, []
                train_lbls_kdef, val_lbls_kdef = kdef_lbls, []
            else:
                train_imgs_kdef, val_imgs_kdef, train_lbls_kdef, val_lbls_kdef = train_test_split(
                    kdef_imgs, kdef_lbls, test_size=0.2, random_state=42
                )
            # Augment KDEF (Same as User Data)
            train_imgs_kdef = augment_offline(train_imgs_kdef, 1)
            
            train_lbls_kdef = [CATEGORIES.index(cat)] * len(train_imgs_kdef)
            val_lbls_kdef = [CATEGORIES.index(cat)] * len(val_imgs_kdef)
        else:
            train_imgs_kdef, val_imgs_kdef = [], []
            train_lbls_kdef, val_lbls_kdef = [], []

        # --- 2. Load FER Data ---
        fer_imgs, fer_lbls = load_images_from_folder(RAW_TRAIN_DIR, cat)
        
        if len(fer_imgs) > 0:
            split_fer = int(len(fer_imgs) * 0.8) 
            train_imgs_fer = fer_imgs[:split_fer]
            val_imgs_fer = fer_imgs[split_fer:]
            
            # Special Handling: Augment FER data for weak classes
            if cat in ["disgust", "fear"]:
                print(f"   ⚠️  [{cat.upper()}] Boosting FER Validation size by {train_mult}x...")
                train_imgs_fer = augment_offline(train_imgs_fer, train_mult) # Moderate boost for FER Train
            
            # Generate labels
            train_lbls_fer = [CATEGORIES.index(cat)] * len(train_imgs_fer)
            val_lbls_fer = [CATEGORIES.index(cat)] * len(val_imgs_fer)
        else:
            train_imgs_fer, val_imgs_fer = [], []
            train_lbls_fer, val_lbls_fer = [], []
        
        # --- 3. Combine ---
        X_train_list.extend(train_imgs_my + train_imgs_kdef + train_imgs_fer)
        y_train_list.extend(train_lbls_my + train_lbls_kdef + train_lbls_fer)
        
        X_val_list.extend(val_imgs_my + val_imgs_kdef + val_imgs_fer)
        y_val_list.extend(val_lbls_my + val_lbls_kdef + val_lbls_fer)
        
        # --- 4. Test Data (Never Augmented) ---
        t_imgs, t_lbls = load_images_from_folder(RAW_TEST_DIR, cat)
        X_test_list.extend(t_imgs)
        y_test_list.extend(t_lbls)

        print(f"     -> {cat.upper()}: {len(train_imgs_my)+len(train_imgs_kdef)+len(train_imgs_fer)} Train | {len(val_imgs_my)+len(val_imgs_kdef)+len(val_imgs_fer)} Val")

    # Arrays & Norm
    print("\n🔄 Converting to Arrays...")
    X_train = np.array(X_train_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    y_train = to_categorical(np.array(y_train_list), num_classes=7)
    
    X_val = np.array(X_val_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    y_val = to_categorical(np.array(y_val_list), num_classes=7)
    
    X_test = np.array(X_test_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    y_test = to_categorical(np.array(y_test_list), num_classes=7)

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_npy(name, data):
    if not os.path.exists(PROCESSED_DIR): os.makedirs(PROCESSED_DIR)
    np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), data)

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    save_npy("X_train", X_train)
    save_npy("y_train", y_train)
    save_npy("X_val", X_val)
    save_npy("y_val", y_val)
    save_npy("X_test", X_test)
    save_npy("y_test", y_test)
    print("✅ Done. Validation data is now balanced.")