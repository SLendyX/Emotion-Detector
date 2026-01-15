import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIG ---
BASE_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"
BATCH_SIZE = 32
EPOCHS = 50 

if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

def build_model():
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.2), # Reduced from 0.3

        # Block 2
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3), # Reduced from 0.4

        # Block 3
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4), # Reduced from 0.5

        # Classifier
        Flatten(),
        # Added L2 Regularization to stop weights from exploding\
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(7, activation='softmax')
    ])
    
    # Lower Learning Rate (0.0005) for stability
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss='categorical_crossentropy', 
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

    # Calculate Class Weights
    y_integers = np.argmax(y_train, axis=1)
    weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    weights_dict = dict(enumerate(weights))

    # --- ADD THIS: Manually boost Sadness ---
    #"angry"=0, "disgust"=1, "fear"=2, "happy"=3, "neutral"=4, "sad"=5, "surprise=6"    
    print(f"⚖️  Class Weights: {weights_dict}")

    # Augmented Generator (Training Only)
    train_datagen = ImageDataGenerator(
        rotation_range=10,       # Gentle rotation
        width_shift_range=0.1,   
        height_shift_range=0.1,  
        shear_range=0.1,         
        zoom_range=0.1,          
        horizontal_flip=True,    
        fill_mode='constant',    
        cval=0                   
    )

    model = build_model()
    
    callbacks = [
        ModelCheckpoint(f"{MODELS_DIR}/best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(
            monitor='val_accuracy',  # Watch accuracy instead of loss
            mode='max',              # We want accuracy to go UP
            patience=10,             # Wait a bit longer
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    print("🚀 Starting Training...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        class_weight=weights_dict,
        callbacks=callbacks
    )

    # Save History
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(RESULTS_DIR, "history.csv"), index=False)
    print("\n✅ Training Complete.")

if __name__ == "__main__":
    main()