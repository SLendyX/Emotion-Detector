import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
# Removed GaussianNoise import
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.losses import CategoricalCrossentropy

# --- CONFIG ---
BASE_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"
BATCH_SIZE = 64
EPOCHS = 70 

if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

def build_model():
    model = Sequential([
        # --- CHANGE: Removed GaussianNoise ---
        # We want maximum clarity so the model can spot the difference 
        # between "wide eyes" (Fear) and "droopy eyes" (Sadness).

        # Block 1
        Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        # Block 2
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # Block 3
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        
        # Block 4
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.4),

        # Classifier
        GlobalAveragePooling2D(),
        # Keep L2 low (0.0001) as it worked well in your best run
        Dense(512, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    # Keep Label Smoothing - it was the key to your success
    model.compile(optimizer=Adam(learning_rate=0.0003), 
                  loss=CategoricalCrossentropy(label_smoothing=0.1), 
                  metrics=['accuracy'])
    return model

def main():
    print("🔄 Loading Data...")
    try:
        X_train = np.load(f"{BASE_DIR}/X_train.npy")
        y_train = np.load(f"{BASE_DIR}/y_train.npy")
        X_val = np.load(f"{BASE_DIR}/X_val.npy")
        y_val = np.load(f"{BASE_DIR}/y_val.npy")
    except FileNotFoundError:
        print("❌ Error: .npy files not found. Run preprocess_data.py first.")
        return

    # --- WEIGHTS STRATEGY ---
    # We stick close to your BEST run (Sadness=1.35), but nudge it slightly.
    # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Neutral, 5=Sadness, 6=Surprise
    manual_weights = {
        0: 1.25, # Angry: Good
        1: 2.0,  # Disgust: Good
        2: 1.15, # Fear: Small boost (1.25 -> 1.35) just to help it compete
        3: 1.0,  # Happy
        4: 1.0,  # Neutral
        5: 1.25, # Sadness: Small trim (1.35 -> 1.25). Not a huge cut, just a "trim".
        6: 1.0   # Surprise
    }

    print(f"⚖️  Using Stabilized Weights: {manual_weights}")

    train_datagen = ImageDataGenerator(
        rotation_range=15, # Reduced slightly from 20 to keep shapes clearer
        width_shift_range=0.1,   
        height_shift_range=0.1,  
        shear_range=0.1,         
        zoom_range=0.1,  
        horizontal_flip=True,    
        fill_mode='nearest'
    )

    model = build_model()
    
    callbacks = [
        ModelCheckpoint(f"{MODELS_DIR}/trained_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    ]

    print("🚀 Starting Training...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        class_weight=manual_weights,
        callbacks=callbacks
    )

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)
    print("\n✅ Training Complete.")

if __name__ == "__main__":
    main()