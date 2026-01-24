import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.losses import CategoricalCrossentropy
from sklearn.metrics import accuracy_score, f1_score

# --- CONFIG ---
BASE_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"
EXPERIMENTS_FILE = os.path.join(RESULTS_DIR, "experiments_results.csv")

if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

# --- LOAD DATA ---
def load_data():
    print("🔄 Loading Data...")
    try:
        X_train = np.load(f"{BASE_DIR}/X_train.npy")
        y_train = np.load(f"{BASE_DIR}/y_train.npy")
        X_val = np.load(f"{BASE_DIR}/X_val.npy")
        y_val = np.load(f"{BASE_DIR}/y_val.npy")
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        print("❌ Error: .npy files not found.")
        return None, None, None, None

# --- MODEL BUILDER FACTORY ---
def build_model(config):
    """
    Constructs a model based on the configuration dictionary.
    config keys: 'dropout_rate', 'l2_reg', 'extra_dense_layer', 'learning_rate'
    """
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(config.get('dropout_1', 0.2)))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(config.get('dropout_2', 0.3)))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(config.get('dropout_3', 0.4)))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(config.get('dropout_4', 0.4)))

    # Classifier
    model.add(GlobalAveragePooling2D())
    
    # Optional Extra Dense Layer (Experiment Architecture)
    if config.get('extra_dense_layer', False):
        model.add(Dense(1024, kernel_regularizer=l2(config.get('l2_reg', 0.0001))))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    model.add(Dense(512, kernel_regularizer=l2(config.get('l2_reg', 0.0001))))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=config.get('learning_rate', 0.0003)), 
        loss=CategoricalCrossentropy(label_smoothing=0.1), 
        metrics=['accuracy']
    )
    return model

def run_experiments():
    X_train, y_train, X_val, y_val = load_data()
    if X_train is None: return

    # --- DEFINE EXPERIMENTS ---
    experiments = {
        "Exp_1_Baseline": {
            "learning_rate": 0.0003,
            "batch_size": 64,
            "dropout_1": 0.2, "dropout_2": 0.3, "dropout_3": 0.4, "dropout_4": 0.4,
            "extra_dense_layer": False,
            "l2_reg": 0.0001
        },
        "Exp_2_HighLR": {
            "learning_rate": 0.001, # Higher LR
            "batch_size": 64,
            "dropout_1": 0.2, "dropout_2": 0.3, "dropout_3": 0.4, "dropout_4": 0.4,
            "extra_dense_layer": False,
            "l2_reg": 0.0001
        },
        "Exp_3_DeepArchitecture": {
            "learning_rate": 0.0003,
            "batch_size": 64,
            "dropout_1": 0.25, "dropout_2": 0.35, "dropout_3": 0.45, "dropout_4": 0.5, # Slightly higher dropout
            "extra_dense_layer": True, # Added layer
            "l2_reg": 0.0001
        },
        "Exp_4_HighReg": {
            "learning_rate": 0.0003,
            "batch_size": 64,
            "dropout_1": 0.3, "dropout_2": 0.4, "dropout_3": 0.5, "dropout_4": 0.5, # Aggressive Dropout
            "l2_reg": 0.01, # Aggressive L2
            "extra_dense_layer": False,
        }
    }

    # Data Augmentation (Standard)
    train_datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )

    results_list = []
    best_val_acc = 0
    best_exp_name = ""

    # Weights Strategy (Shared)
    manual_weights = {0: 1.25, 1: 2.0, 2: 1.35, 3: 1.0, 4: 1.0, 5: 1.25, 6: 1.0}

    print(f"🧪 Starting {len(experiments)} Experiments...")

    for exp_name, config in experiments.items():
        print(f"\n▶ Running {exp_name}...")
        
        model = build_model(config)
        
        # Callbacks (Short patience for experiments)
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]

        # Train
        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=config['batch_size']),
            validation_data=(X_val, y_val),
            epochs=35, # Reduced epochs for experimentation speed (vs 70 for final)
            class_weight=manual_weights,
            callbacks=callbacks,
            verbose=2
        )

        # Evaluate
        val_pred_probs = model.predict(X_val, verbose=0)
        val_pred_classes = np.argmax(val_pred_probs, axis=1)
        val_true_classes = np.argmax(y_val, axis=1)

        acc = accuracy_score(val_true_classes, val_pred_classes)
        f1 = f1_score(val_true_classes, val_pred_classes, average='macro')
        
        print(f"✅ {exp_name} Result: Val Acc={acc:.4f}, Val F1={f1:.4f}")
        
        # Save Result
        results_list.append({
            "Experiment": exp_name,
            "Accuracy": acc,
            "F1_Score": f1,
            "Epochs": len(history.history['loss']),
            "Config": str(config)
        })

        # Save Best Model Logic
        if acc > best_val_acc:
            best_val_acc = acc
            best_exp_name = exp_name
            print(f"🏆 New Best Model found! Saving to {MODELS_DIR}/optimized_model.h5")
            model.save(f"{MODELS_DIR}/optimized_model.h5")

    # Save Results Table
    df = pd.DataFrame(results_list)
    df.to_csv(EXPERIMENTS_FILE, index=False)
    print(f"\n📊 Experiments Complete. Results saved to {EXPERIMENTS_FILE}")
    print(f"🌟 Best Experiment: {best_exp_name} with Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    run_experiments()
