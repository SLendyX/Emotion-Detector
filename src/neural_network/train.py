import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURARE MANIFEST ---
BASE_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"
DOCS_DIR = "docs"

# Creare directoare dacă nu există
for directory in [MODELS_DIR, RESULTS_DIR, DOCS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

BATCH_SIZE = 32
EPOCHS = 60 # Am crescut ușor limita, EarlyStopping va opri oricum când e gata

def build_model():
    # Arhitectură îmbunătățită "VGG-style" pentru a trece de 65%
    model = Sequential([
        # Bloc 1 - Detalii fine (linii, colțuri)
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        # Bloc 2 - Forme medii (ochi, gură)
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),

        # Bloc 3 - Concepte complexe (emoții subtile) - NOU ADĂUGAT
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.5),

        # Clasificator
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    
    # Learning rate 0.001 este standard, scheduler-ul îl va scădea singur
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def save_training_plots(history):
    """Salvează graficele în docs/loss_curve.png"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))
    
    # Grafic Acuratețe
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Grafic Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.savefig(os.path.join(DOCS_DIR, "loss_curve.png"))
    # plt.show() # Poți comenta asta dacă rulezi pe server fără ecran
    print(f"✅ Grafice salvate în {DOCS_DIR}/loss_curve.png")

def main():
    print("🔄 Încărcare date procesate...")
    # 1. Încărcare date
    try:
        X_train = np.load(f"{BASE_DIR}/X_train.npy")
        y_train = np.load(f"{BASE_DIR}/y_train.npy")
        X_val = np.load(f"{BASE_DIR}/X_val.npy")
        y_val = np.load(f"{BASE_DIR}/y_val.npy")
    except FileNotFoundError:
        print("❌ Eroare: Nu găsesc fișierele .npy. Rulează întâi data_cleaner.py!")
        return

    # 2. Balansare clase (Class Weights)
    # Important pentru că FER2013 este natural nebalansat
    y_integers = np.argmax(y_train, axis=1)
    weights = compute_class_weight('balanced', classes=np.unique(y_integers), y=y_integers)
    weights_dict = dict(enumerate(weights))
    print(f"⚖️  Class Weights calculate: {weights_dict}")

    # 3. Augmentare Avansată (Nivel 2)
    # Critică pentru a evita memorarea feței tale
    datagen = ImageDataGenerator(
        rotation_range=20,       # Rotire ușoară
        width_shift_range=0.1,   # Deplasare stânga-dreapta
        height_shift_range=0.1,  # Deplasare sus-jos
        shear_range=0.1,         # Deformare ușoară
        zoom_range=0.2,          # Zoom (important pentru distanța față de cameră)
        horizontal_flip=True,    # Oglindire
        fill_mode='nearest'
    )

    model = build_model()
    model.summary()

    # 4. Callbacks
    callbacks = [
        ModelCheckpoint(f"{MODELS_DIR}/trained_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001, verbose=1)
    ]

    print("🚀 Start antrenare...")
    # 5. Antrenare
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val), # Validare pe setul mixt
        epochs=EPOCHS,
        class_weight=weights_dict,
        callbacks=callbacks
    )

    # 6. Salvare rezultate
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(RESULTS_DIR, "training_history.csv"), index=False)
    
    save_training_plots(history)
    
    print("\n✅ Antrenare completă. Modelul este salvat în models/trained_model.h5")

if __name__ == "__main__":
    main()