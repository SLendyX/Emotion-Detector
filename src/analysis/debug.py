import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model

# Config
BASE_DIR = "data/processed"
MODELS_DIR = "models"
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def show_errors():
    # Load data and model
    X_val = np.load(f"{BASE_DIR}/X_val.npy")
    y_val = np.load(f"{BASE_DIR}/y_val.npy")
    model = load_model(f"{MODELS_DIR}/best_model.keras")

    # Get predictions
    predictions = model.predict(X_val)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_val, axis=1)

    # Find "Neutral" (True) -> "Angry" (Predicted) errors
    neutral_idx = CATEGORIES.index("neutral")
    angry_idx = CATEGORIES.index("angry")
    
    # Get indices of mistakes
    errors = np.where((true_classes == neutral_idx) & (pred_classes == angry_idx))[0]
    
    print(f"Found {len(errors)} Neutral faces mistaken for Angry.")
    
    if len(errors) > 0:
        plt.figure(figsize=(10, 5))
        for i, idx in enumerate(errors[:10]): # Show top 10 errors
            plt.subplot(2, 5, i+1)
            plt.imshow(X_val[idx].reshape(48, 48), cmap='gray')
            plt.title(f"Pred: Angry\nTrue: Neutral")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    show_errors()