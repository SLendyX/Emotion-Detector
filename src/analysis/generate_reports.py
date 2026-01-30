import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np
import os
import glob
import random
import pandas as pd

# Config
MODEL_PATH = 'models/optimized_model.pt'
TEST_DIR = 'data/raw/test'
DOCS_DIR = 'docs/results'
SCREENSHOTS_DIR = 'docs/screenshots'
OPTIMIZATION_DIR = 'docs/optimization'
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

# Dataset Simplu pt Test
class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = glob.glob(os.path.join(root_dir, '*', '*.jpg'))
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        label_str = os.path.basename(os.path.dirname(path))
        # Handle case sensitivity
        label_str = label_str.capitalize() 
        if label_str not in self.class_to_idx:
            # Fallback search
            for c in CLASSES:
                if c.lower() == label_str.lower():
                    label_str = c
                    break
        
        label = self.class_to_idx[label_str]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label, path

def load_robust_model():
    print(f"🔄 Loading Model from {MODEL_PATH}...")
    model = models.resnet18(weights=None)
    
    # 1. Încercăm arhitectura standard (Sequential: Dropout -> Linear)
    # Asta corespunde experimentelor Exp 1, 2, 4 din run_experiments.py
    try:
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, len(CLASSES))
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Model loaded (Standard Architecture)")
    except RuntimeError:
        # 2. Încercăm arhitectura Deep (Exp 3)
        try:
            model.fc = nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, len(CLASSES))
            )
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("✅ Model loaded (Deep Architecture)")
        except RuntimeError:
            # 3. Fallback la simplu Linear (dacă modelul a fost salvat diferit)
            model.fc = nn.Linear(512, len(CLASSES))
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("✅ Model loaded (Simple Linear)")
            
    model.to(DEVICE)
    model.eval()
    return model

def generate_comparisons():
    print("📊 Generare grafice comparative...")
    exp_path = 'results/old_results/experiments_results.csv'
    if not os.path.exists(exp_path):
        # Încearcă calea alternativă
        exp_path = 'results/experiments_results.csv' 
        if not os.path.exists(exp_path):
            print(f"⚠️ Warning: {exp_path} not found. Skipping comparisons.")
            return

    df = pd.read_csv(exp_path)
    
    # Accuracy Comparison
    plt.figure(figsize=(10, 6))
    if 'Accuracy' in df.columns:
        sns.barplot(x='Experiment', y='Accuracy', data=df, palette='viridis')
        plt.title('Accuracy per Experiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OPTIMIZATION_DIR, 'accuracy_comparison.png'))
        plt.close()

    # F1 Comparison
    plt.figure(figsize=(10, 6))
    if 'F1_Score' in df.columns:
        sns.barplot(x='Experiment', y='F1_Score', data=df, palette='magma')
        plt.title('F1-Score per Experiment')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OPTIMIZATION_DIR, 'f1_comparison.png'))
        plt.close()
    print("✅ Grafice comparative salvate.")

def generate_learning_curves():
    print("📈 Generare curbe de învățare...")
    hist_path = 'results/history.csv'
    if not os.path.exists(hist_path):
        print(f"⚠️ Warning: {hist_path} not found. Skipping learning curves.")
        return

    df = pd.read_csv(hist_path)
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df['train_acc'], label='Train Accuracy')
    plt.plot(df['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(df['train_loss'], label='Train Loss')
    plt.plot(df['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OPTIMIZATION_DIR, 'learning_curves_best.png'))
    plt.close()
    print("✅ Curbe de învățare salvate.")

def main():
    print("📊 Începere Generare Rapoarte...")
    
    generate_comparisons()
    generate_learning_curves()
    
    # 1. Load Model
    model = load_robust_model()

    # 2. Inference
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = SimpleDataset(TEST_DIR, transform)
        if len(dataset) == 0:
            print(f"❌ Error: Nu s-au găsit imagini în {TEST_DIR}")
            return
            
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    all_preds = []
    all_labels = []
    error_examples = [] # Store (path, true, pred)

    print("   Running inference...")
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            _, preds = torch.max(out, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Colectare erori
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    error_examples.append((paths[i], CLASSES[labels[i]], CLASSES[preds[i]]))

    # 3. Metrici
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"✅ Accuracy Final: {acc*100:.2f}%")
    print(f"✅ F1-Score Macro: {f1:.4f}")
    
    # 4. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix (Acc={acc:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(DOCS_DIR, 'confusion_matrix_optimized.png'))
    plt.close()
    
    # 5. Salvare 5 Exemple Greșite
    print("🖼️ Salvare 5 exemple greșite...")
    if error_examples:
        random.shuffle(error_examples)
        fig, axes = plt.subplots(1, 5, figsize=(15, 4))
        
        for i in range(min(5, len(error_examples))):
            path, true_lbl, pred_lbl = error_examples[i]
            img = Image.open(path)
            axes[i].imshow(img)
            axes[i].set_title(f"True: {true_lbl}\nPred: {pred_lbl}", color='red')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(SCREENSHOTS_DIR, 'error_examples_grid.png'))
        plt.close()
    else:
        print("   Felicitări! Nicio eroare găsită.")
        
    print("✅ Rapoarte generate cu succes!")

if __name__ == "__main__":
    main()