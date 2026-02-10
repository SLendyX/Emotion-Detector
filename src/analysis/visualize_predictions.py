import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# --- CONFIGURARE ---
# Calea către modelul cel mai bun (modifică dacă numele e diferit)
MODEL_PATH = "models/experiments/exp1_best.pt" 
TEST_DIR = "data/raw/test"
OUTPUT_DIR = "docs/grafice"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "example_predictions.png")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definirea claselor (în ordinea folderelor)
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# --- DEFINIREA MODELULUI (Trebuie să fie identică cu cea din antrenare) ---
class SimpleEmotionCNN(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.0):
        super(SimpleEmotionCNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(128 * 12 * 12, num_classes)

    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(x)))
        out = out.view(out.size(0), -1)
        if self.drop.p > 0: out = self.drop(out)
        out = self.fc(out)
        return out

# --- FUNCȚII AUXILIARE ---
def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        # Normalizarea folosita la antrenare
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def denormalize(tensor):
    """Inversează normalizarea pentru a putea afișa imaginea corect cu matplotlib"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convertim din Tensor [C, H, W] în Numpy [H, W, C]
    img_np = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Denormalizare: img = img * std + mean
    img_np = img_np * std + mean
    
    # Ne asigurăm că valorile sunt între 0 și 1
    img_np = np.clip(img_np, 0, 1)
    return img_np

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Eroare: Nu am găsit modelul la {MODEL_PATH}")
        return

    # 1. Încărcare Date și Model
    val_tf = get_val_transforms()
    # Folosim ImageFolder care citeste automat structura folderelor ca etichete
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_tf)
    # Luăm un batch mai mare ca să avem de unde alege
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True) 

    print(f"Încărcare model din: {MODEL_PATH}")
    # Dropout rate nu conteaza la evaluare, dar trebuie sa initializam clasa
    model = SimpleEmotionCNN(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Important: dezactivează dropout și batchnorm

    # 2. Colectare Predicții
    correct_samples = []
    incorrect_samples = []

    print("Rulez inferența pentru a găsi exemple...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(images)):
                img_tensor = images[i]
                true_idx = labels[i].item()
                pred_idx = preds[i].item()
                
                sample_data = (img_tensor, true_idx, pred_idx)
                
                if true_idx == pred_idx:
                    correct_samples.append(sample_data)
                else:
                    incorrect_samples.append(sample_data)
            
            # Ne oprim dacă avem suficiente exemple din ambele categorii
            if len(correct_samples) > 20 and len(incorrect_samples) > 20:
                break
    
    print(f"Găsit: {len(correct_samples)} corecte, {len(incorrect_samples)} greșite.")

    # 3. Selecție Mixată (ex: 5 corecte, 4 greșite pentru un grid de 9)
    n_correct_to_show = min(len(correct_samples), 5)
    n_incorrect_to_show = min(len(incorrect_samples), 9 - n_correct_to_show)
    
    final_samples = random.sample(correct_samples, n_correct_to_show) + \
                    random.sample(incorrect_samples, n_incorrect_to_show)
    
    # Amestecăm lista finală ca să nu fie grupate
    random.shuffle(final_samples)

    # 4. Plotting Grid 3x3
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 11))
    fig.suptitle('Exemple Predicții Model Optimizat (Verde=Corect, Roșu=Greșit)', fontsize=14, y=0.99)
    
    for i, ax in enumerate(axes.flat):
        if i < len(final_samples):
            img_tensor, true_idx, pred_idx = final_samples[i]
            
            # Pregătire imagine pentru afișare
            img_to_plot = denormalize(img_tensor)
            ax.imshow(img_to_plot)
            
            true_label = class_names[true_idx]
            pred_label = class_names[pred_idx]
            
            if true_idx == pred_idx:
                color = 'green'
                title = f"{true_label}"
            else:
                color = '#d63031' # Un roșu mai plăcut
                # Afișăm ce a prezis vs ce trebuia să fie
                title = f"Pred: {pred_label}\nReal: {true_label}"
                
            ax.set_title(title, color=color, fontsize=11, fontweight='bold')
        
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"✅ Grid salvat în: {OUTPUT_FILE}")
    # plt.show()

if __name__ == "__main__":
    main()