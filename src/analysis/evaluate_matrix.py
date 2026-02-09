import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import os
import sys

# --- FIX IMPORTURI (Adăugăm folderul 'src' la calea Python) ---
# Obținem calea către folderul curent (src/analysis)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Obținem folderul părinte (src)
src_dir = os.path.dirname(current_dir)
# Obținem rădăcina proiectului (Proiect/Emotion-Detector) - pentru date
project_root = os.path.dirname(src_dir)

sys.path.append(src_dir)
sys.path.append(project_root)

# Acum importul va funcționa fără ".." (puncte)
# Presupunând că my_training.py este în src/neural_network/
from neural_network.my_training import SimpleEmotionCNN, EmotionDataset, class_map

# --- CONFIG ACTUALIZAT (Căi absolute pentru a evita erori de fișiere) ---
# Folosim project_root pentru a construi căile corecte, indiferent de unde rulezi scriptul
MODEL_PATH = os.path.join(project_root, "models/emotion_model_epoch_50.pt")
TEST_DIR = os.path.join(project_root, "data/raw/test")
SAVE_DIR = os.path.join(project_root, "docs") # Folderul unde salvăm graficele

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapare inversă (0 -> angry)
idx_to_class = {v: k for k, v in class_map.items()}
class_names = [idx_to_class[i] for i in range(7)]

def main():
    # 1. Setup Date și Model
    val_transforms = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Folosim doar setul de testare
    test_set = EmotionDataset(raf_dir=TEST_DIR, gen_dir="", transform=val_transforms)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleEmotionCNN(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []
    mistakes = [] # Vom stoca imaginile greșite aici

    print("🔍 Rulare inferență pe setul de test...")
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Colectăm exemple greșite (doar primele 5)
            if len(mistakes) < 5:
                wrong_indices = (preds != labels).nonzero()
                for idx in wrong_indices:
                    if len(mistakes) < 5:
                        img_idx = idx.item()
                        # Denormalizare pentru afișare corectă
                        img = images[img_idx].cpu().permute(1, 2, 0).numpy()
                        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
                        img = np.clip(img, 0, 1)
                        
                        mistakes.append({
                            'img': img,
                            'true': class_names[labels[img_idx].item()],
                            'pred': class_names[preds[img_idx].item()]
                        })

    # 2. Generare Matrice de Confuzie
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicție Model')
    plt.ylabel('Etichetă Reală')
    plt.title('Confusion Matrix')
    plt.savefig('docs/confusion_matrix.png')
    print("✅ Matricea salvată în docs/confusion_matrix.png")

    # 3. Afișare 5 Exemple Greșite
    plt.figure(figsize=(15, 5))
    for i, item in enumerate(mistakes):
        plt.subplot(1, 5, i+1)
        plt.imshow(item['img'])
        plt.title(f"Real: {item['true']}\nPred: {item['pred']}", color='red')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/failed_examples.png')
    print("✅ Exemplele greșite salvate în docs/failed_examples.png")

if __name__ == "__main__":
    main()