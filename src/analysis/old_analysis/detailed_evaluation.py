import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# --- CONFIG ---
MODEL_PATH = "models/optimized_model.h5"
DATA_PATH = "data/processed/X_test_clean.npy"
LABEL_PATH = "data/processed/y_test_clean.npy"
DOCS_DIR = "docs"
SCREENSHOTS_DIR = os.path.join(DOCS_DIR, "screenshots")
CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise"]

if not os.path.exists(SCREENSHOTS_DIR): os.makedirs(SCREENSHOTS_DIR)

def evaluate_detailed():
    print(f"🔄 Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
    except OSError:
        print("❌ Model file not found. Please run experiments first.")
        return

    print("🔄 Loading Test Data...")
    X_test = np.load(DATA_PATH)
    y_test = np.load(LABEL_PATH)

    # Predictions
    print("🔮 Running Inference...")
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # --- 1. METRICS ---
    report = classification_report(y_true_classes, y_pred_classes, target_names=CATEGORIES, output_dict=True)
    print("\n📊 --- FINAL METRICS ON TEST SET ---")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"F1-Score (Macro): {report['macro avg']['f1-score']:.4f}")

    # --- 2. CONFUSION MATRIX ---
    print("\n🎨 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title('Confusion Matrix - Optimized Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(DOCS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"✅ Saved to {cm_path}")

    # --- 3. MISCLASSIFIED EXAMPLES ---
    print("\n🔍 Finds 5 Misclassified Examples...")
    misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
    
    # Select 5 random errors
    if len(misclassified_indices) > 5:
        selected_indices = np.random.choice(misclassified_indices, 5, replace=False)
    else:
        selected_indices = misclassified_indices

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, idx in enumerate(selected_indices):
        img = X_test[idx].reshape(48, 48)
        true_label = CATEGORIES[y_true_classes[idx]]
        pred_label = CATEGORIES[y_pred_classes[idx]]
        
        # Save individual error image for documentation
        # Un-normalize for saving: x * 255
        img_save = (img * 255).astype(np.uint8)
        error_filename = os.path.join(SCREENSHOTS_DIR, f"error_example_{i+1}_{true_label}_pred_{pred_label}.png")
        cv2.imwrite(error_filename, img_save)

        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color='red')
        axes[i].axis('off')
    
    errors_plot_path = os.path.join(SCREENSHOTS_DIR, "misclassified_batch.png")
    plt.tight_layout()
    plt.savefig(errors_plot_path)
    plt.close()
    print(f"✅ Saved 5 error examples to {SCREENSHOTS_DIR}")

if __name__ == "__main__":
    evaluate_detailed()
