import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Adjust this path if your history.npy is somewhere else
HISTORY_PATH = "models/history.npy" 

def plot_metrics(history_path):
    if not os.path.exists(history_path):
        print(f"❌ Error: Could not find file at {history_path}")
        print("   Make sure you ran train.py first!")
        return

    # Load the history file
    # allow_pickle=True is needed because history is a dictionary, not just numbers
    try:
        history = np.load(history_path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ Error loading .npy file: {e}")
        return

    # Extract data
    acc = history.get('accuracy', [])
    val_acc = history.get('val_accuracy', [])
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    
    epochs = range(1, len(acc) + 1)

    # --- PLOT 1: ACCURACY (How smart is it?) ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Acc')
    plt.plot(epochs, val_acc, 'r*-', label='Validation Acc (Test)')
    plt.title('Accuracy: Training vs. Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # --- PLOT 2: LOSS (How confused is it?) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss (Test)')
    plt.title('Loss: Training vs. Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_metrics(HISTORY_PATH)