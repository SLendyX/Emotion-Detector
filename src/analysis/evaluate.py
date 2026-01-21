import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Configurare path-uri conform README
MODEL_PATH = "models/trained_model.h5"
DATA_PATH = "data/processed/X_test.npy"
LABEL_PATH = "data/processed/y_test.npy"
RESULTS_DIR = "results"
DOCS_DIR = "docs"
CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

def evaluate():
    # 1. Încărcare model și date
    model = load_model(MODEL_PATH)
    X_test = np.load(DATA_PATH)
    y_test = np.load(LABEL_PATH)
    
    print(len(X_test))
    
    # 2. Predicții
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # 3. Calcul Metricii (Nivel 1 - Accuracy & F1)
    report = classification_report(y_true, y_pred_classes, target_names=CATEGORIES, output_dict=True)
    
    metrics = {
        "test_accuracy": report["accuracy"],
        "test_f1_macro": report["macro avg"]["f1-score"],
        "test_precision_macro": report["macro avg"]["precision"],
        "test_recall_macro": report["macro avg"]["recall"]
    }
    
    # Salvare JSON
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    # 4. Confusion Matrix (Nivel 3 Bonus)
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.xlabel('Predicție')
    plt.ylabel('Adevărat')
    plt.savefig(os.path.join(DOCS_DIR, "confusion_matrix.png"))
    
    print(f"✅ Evaluare finalizată! Accuracy: {metrics['test_accuracy']:.2f}\n F1: {metrics['test_f1_macro']:.2f}\n Precision: {metrics["test_precision_macro"]:.2f}\n Recall:{metrics["test_recall_macro"]:.2f}")

if __name__ == "__main__":
    evaluate()