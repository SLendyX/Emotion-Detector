import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight # <--- NOU 1: Importăm funcția

# --- CONFIGURARE ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = "data/processed"
MODELS_DIR = "models"
DOCS_DIR = "docs"

BATCH_SIZE = 32
EPOCHS = 60
NUM_CLASSES = 7
IMG_SIZE = 48

def load_data():
    print("🔄 Încărcare date din .npy...")
    try:
        X_train = np.load(os.path.join(BASE_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(BASE_DIR, "y_train.npy"))
        X_test = np.load(os.path.join(BASE_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(BASE_DIR, "y_test.npy"))
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print("❌ EROARE: Nu găsesc fișierele .npy.")
        return None, None, None, None

def build_model(input_shape):
    model = Sequential([
        # Bloc 1
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Bloc 2
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Bloc 3 - Clasificare
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    
    return model

def plot_and_save_history(history):
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)

    save_path = os.path.join(DOCS_DIR, "grafice_antrenare.png")
    plt.savefig(save_path)
    plt.close()

def main():
    X_train, y_train, X_test, y_test = load_data()
    if X_train is None: return

    # --- NOU 2: Calculăm Class Weights ---
    # y_train este One-Hot (ex: [0,0,1,0...]). Trebuie să îl facem numere simple (0, 1, 2)
    y_integers = np.argmax(y_train, axis=1)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    # Transformăm într-un dicționar pentru Keras: {0: 1.0, 1: 0.8, ...}
    class_weights_dict = dict(enumerate(class_weights))
    
    print("\n⚖️  Greutăți calculate pentru balansare:")
    for i, w in class_weights_dict.items():
        print(f"   Clasa {i}: {w:.4f}")

    # Model Building
    model = build_model(X_train.shape[1:])
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    checkpoint = ModelCheckpoint(os.path.join(MODELS_DIR, 'emotion_model.keras'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    print("\n🚀 Începem antrenarea cu Class Weights...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights_dict, # <--- NOU 3: Aici pasăm greutățile
        shuffle=True
    )

    np.save(os.path.join(MODELS_DIR, 'history.npy'), history.history)
    plot_and_save_history(history)
    print("\n✅ Antrenare finalizată!")

if __name__ == "__main__":
    main()