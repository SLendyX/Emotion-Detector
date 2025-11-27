import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# --- CONFIGURARE ---
DATA_DIR = "data/processed"
MODELS_DIR = "models"
BATCH_SIZE = 64
EPOCHS = 50  # Numărul de treceri prin tot setul de date
NUM_CLASSES = 7  # Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

def build_model(input_shape):
    """
    Definim arhitectura CNN.
    Este o arhitectură stil VGG (blocuri de convoluție urmate de pooling).
    """
    model = Sequential()

    # --- Bloc 1: Detecție trăsături de bază ---
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization()) # Normalizează datele intern pentru viteză
    model.add(MaxPooling2D(pool_size=(2, 2))) # Micșorează imaginea la jumătate
    model.add(Dropout(0.25)) # "Uită" aleatoriu 25% din neuroni (evită tocitul/overfitting)

    # --- Bloc 2: Detecție trăsături medii (ochi, gură) ---
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # --- Bloc 3: Clasificare (Creierul decizional) ---
    model.add(Flatten()) # Transformă matricea 3D în vector 1D
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5)) # Dropout agresiv înainte de final
    model.add(Dense(NUM_CLASSES, activation='softmax')) # Softmax ne dă probabilitățile (suma lor = 1)

    # Compilarea modelului
    # Folosim Adam (cel mai bun optimizer general) și Categorical Crossentropy (pentru clasificare multiplă)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    
    return model

def main():
    # 1. Încărcăm datele procesate
    print("🔄 Încărcare date din .npy...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    except FileNotFoundError:
        print("❌ Nu am găsit fișierele .npy. Rulează preprocessing-ul întâi!")
        return

    # Verificăm forma datelor (trebuie să fie ex: (28000, 48, 48, 1))
    print(f"Dimensiune Train: {X_train.shape}")
    input_shape = X_train.shape[1:] # (48, 48, 1)

    # 2. Construim modelul
    model = build_model(input_shape)
    model.summary() # Afișează structura în consolă

    # 3. Pregătim callback-urile (Salvări automate)
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    # Salvăm DOAR cel mai bun model (care are cea mai mică eroare pe setul de test)
    checkpoint = ModelCheckpoint(
        os.path.join(MODELS_DIR, 'emotion_model.keras'), # extensia nouă keras
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Oprim antrenarea dacă nu mai învață nimic timp de 7 epoci (economie de timp)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    # 4. START ANTRENARE 🚀
    print("\n🚀 Începem antrenarea! Ia-ți o cafea, durează...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop],
        shuffle=True
    )

    print("\n✅ Antrenare finalizată.")
    
    # Putem salva și istoricul pentru a plota graficele mai târziu
    np.save(os.path.join(MODELS_DIR, 'history.npy'), history.history)

if __name__ == "__main__":
    main()