import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, img_to_array

# --- CONFIGURATION ---
BASE_DIR = "data"
RAW_TRAIN_DIR = os.path.join(BASE_DIR, "raw/train") 
RAW_TEST_DIR = os.path.join(BASE_DIR, "raw/test")   
GENERATED_DIR = os.path.join(BASE_DIR, "generated") # Your 20 raw photos per class
PROCESSED_DIR = os.path.join(BASE_DIR, "processed") 

IMG_SIZE = 48
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
AUGMENT_FACTOR = 20 # 1 photo -> 20 photos

# Define the generator for offline augmentation
aug_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment_offline(images):
    """
    Takes a list of raw images and returns a list of augmented versions.
    """
    augmented_images = []
    
    for img in images:
        # 1. Add original
        augmented_images.append(img)
        
        # 2. Reshape for Keras (48, 48) -> (1, 48, 48, 1)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        
        # 3. Generate variations
        i = 0
        for batch in aug_datagen.flow(x, batch_size=1):
            # Extract image from batch
            aug_img = batch[0].astype('uint8')
            # Remove the extra channel for storage consistency if needed, 
            # but usually we process as arrays. Let's keep it simple:
            aug_img = aug_img.reshape(IMG_SIZE, IMG_SIZE)
            
            augmented_images.append(aug_img)
            i += 1
            if i >= AUGMENT_FACTOR:
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
    
    if limit:
        file_names = file_names[:int(limit)]
        
    class_num = CATEGORIES.index(category)
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    
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
    
    print(f"📊 Processing Data with {AUGMENT_FACTOR}x Augmentation...")

    for cat in CATEGORIES:
        # --- 1. Load YOUR Data ---
        my_imgs, my_lbls = load_images_from_folder(GENERATED_DIR, cat)
        
        # Split FIRST (Crucial for safety)
        # If you have 20 imgs: 16 Train, 4 Val
        if len(my_imgs) >= 5:
            train_imgs_my, val_imgs_my, train_lbls_my, val_lbls_my = train_test_split(
                my_imgs, my_lbls, test_size=0.2, random_state=42
            )
        else:
            # Fallback for tiny classes
            train_imgs_my, val_imgs_my = my_imgs, []
            train_lbls_my, val_lbls_my = my_lbls, []

        # --- 2. AUGMENT Only the Training Split ---
        # 16 imgs -> 336 imgs
        print(f"   [{cat}] Augmenting {len(train_imgs_my)} user photos to {len(train_imgs_my)*(AUGMENT_FACTOR+1)}...")
        train_imgs_my = augment_offline(train_imgs_my)
        # We need to extend the labels to match the new count
        train_lbls_my = [train_lbls_my[0]] * len(train_imgs_my) 
        
        # --- 3. Load FER Data to Match Volume ---
        # Now we have ~300 user photos, so we can grab 600 FER photos
        limit_fer = 600 
        fer_imgs, fer_lbls = load_images_from_folder(RAW_TRAIN_DIR, cat, limit=limit_fer)
        
        split_fer = int(len(fer_imgs) * 0.9) # 10% Val for FER is enough
        train_imgs_fer = fer_imgs[:split_fer]
        train_lbls_fer = fer_lbls[:split_fer]
        val_imgs_fer = fer_imgs[split_fer:]
        val_lbls_fer = fer_lbls[split_fer:]
        
        # --- 4. Combine ---
        X_train_list.extend(train_imgs_my + train_imgs_fer)
        y_train_list.extend(train_lbls_my + train_lbls_fer)
        
        X_val_list.extend(val_imgs_my + val_imgs_fer)
        y_val_list.extend(val_lbls_my + val_lbls_fer)
        
        # Test Data
        t_imgs, t_lbls = load_images_from_folder(RAW_TEST_DIR, cat, limit=100)
        X_test_list.extend(t_imgs)
        y_test_list.extend(t_lbls)

        print(f"     -> Final: {len(train_imgs_my)+len(train_imgs_fer)} Train | {len(val_imgs_my)+len(val_imgs_fer)} Val")

    # Arrays & Norm
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
    print("\n✅ Processing Complete. Data volume restored safely.")