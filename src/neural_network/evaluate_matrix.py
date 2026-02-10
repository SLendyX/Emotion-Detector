import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import sys

# --- FIX IMPORTURI (Adăugăm folderul 'src' la calea Python) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

sys.path.append(src_dir)
sys.path.append(project_root)

from neural_network.train import SimpleEmotionCNN, EmotionDataset, class_map

# --- CONFIG ACTUALIZAT ---
MODEL_PATH = os.path.join(project_root, "models/optimized_model.pt")
TEST_DIR = os.path.join(project_root, "data/raw/test")
SAVE_DIR = os.path.join(project_root, "docs") 

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

idx_to_class = {v: k for k, v in class_map.items()}
class_names = [idx_to_class[i] for i in range(7)]

def main():
    # 1. Setup Date și Model
    val_transforms = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_set = EmotionDataset(raf_dir=TEST_DIR, gen_dir="", transform=val_transforms)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleEmotionCNN(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    mistakes = []

    print("🔍 Rulare inferență pe setul de test...")
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if len(mistakes) < 5:
                wrong_indices = (preds != labels).nonzero()
                for idx in wrong_indices:
                    if len(mistakes) < 5:
                        img_idx = idx.item()
                        img = images[img_idx].cpu().permute(1, 2, 0).numpy()
                        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                        img = np.clip(img, 0, 1)
                        
                        mistakes.append({
                            'img': img,
                            'true': class_names[labels[img_idx].item()],
                            'pred': class_names[preds[img_idx].item()],
                            'conf': probs[img_idx, preds[img_idx].item()].item()
                        })

    # 2. Generare Matrice de Confuzie (CU PROCENTE)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalizare pe rânduri (Row Normalization)
    # Împărțim fiecare valoare la suma rândului (totalul exemplelor reale din acea clasă)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    # fmt='.2%' formatează numărul ca procent cu 2 zecimale (ex: 0.85 -> 85.00%)
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicție Model')
    plt.ylabel('Etichetă Reală')
    plt.title('Confusion Matrix (Normalized by Row %)')
    
    save_path_cm = os.path.join(SAVE_DIR, 'confusion_matrix_percent.png')
    plt.savefig(save_path_cm)
    print(f"✅ Matricea salvată în {save_path_cm}")

    # 3. Afișare 5 Exemple Greșite
    plt.figure(figsize=(15, 5))
    for i, item in enumerate(mistakes):
        plt.subplot(1, 5, i+1)
        plt.imshow(item['img'])
        plt.title(f"Real: {item['true']}\nPred: {item['pred']}\nConf: {item['conf']:.2%}", color='red')
        plt.axis('off')
    
    plt.tight_layout()
    save_path_ex = os.path.join(SAVE_DIR, 'failed_examples.png')
    plt.savefig(save_path_ex)
    print(f"✅ Exemplele greșite salvate în {save_path_ex}")

if __name__ == "__main__":
    main()