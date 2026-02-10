import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import time

# --- CONFIGURATION ---
DATA = "data"
GENERATED_DIR = "data/generated"
TRAIN_DIR = os.path.join(DATA, "raw/train")
TEST_DIR =  os.path.join(DATA, "raw/test")
RESULTS_DIR = "results"
DOCS_DIR = "docs/grafice"

# Numele fișierelor specifice pentru ResNet (ca să nu suprascriem modelul tău)
MODEL_SAVE_PATH = "models/resnet18_best.pt"
HISTORY_SAVE_PATH = "results/resnet_history.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

class_map = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3, 
    "neutral": 4, "sad": 5, "surprised": 6
}

# --- DATASET (Identic cu cel vechi) ---
class EmotionDataset(Dataset):
    def __init__(self, raf_dir, gen_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        def load_images_from_folder(directory):
            if not os.path.exists(directory): return
            for root, dirs, files in os.walk(directory):
                folder_name = os.path.basename(root)
                if folder_name in class_map:
                    for name in files:
                        if name.endswith(('.jpg', '.png', '.jpeg')):    
                            self.image_paths.append(os.path.join(root, name))
                            self.labels.append(class_map.get(folder_name))

        load_images_from_folder(raf_dir)
        load_images_from_folder(gen_dir)

    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

# --- TRANSFORMS ---
def get_transforms():
    # ResNet se descurcă bine cu 224x224, dar pentru comparație corectă păstrăm 100x100
    # sau putem mări la 224 pentru a-i da un avantaj corect ResNet-ului. 
    # Să păstrăm 100x100 ca să fie "fair fight" pe input.
    train_transforms = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms, val_transforms

# --- MODEL RESNET ---
def get_resnet_model():
    print("⬇️ Downloading/Loading ResNet18 pretrained weights...")
    # Încărcăm modelul pre-antrenat
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modificăm ultimul strat (Fully Connected) pentru a avea 7 ieșiri (clasele noastre)
    # ResNet original are 1000 de clase (ImageNet)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    
    return model

# --- MAIN ---
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training ResNet18 on: {device}")

    # Datasets
    train_tf, val_tf = get_transforms()
    train_set = EmotionDataset(TRAIN_DIR, GENERATED_DIR, transform=train_tf)
    val_set = EmotionDataset(TEST_DIR, "", transform=val_tf)

    #Weight Calculation
    raf_indices = [i for i, p in enumerate(train_set.image_paths) if "generated" not in p]
    gen_indices = [i for i, p in enumerate(train_set.image_paths) if "generated" in p]
    w_gen = 0.4 / (len(gen_indices) or 1)
    w_raf = 0.6 / (len(raf_indices) or 1)
    weights = [w_gen if "generated" in p else w_raf for p in train_set.image_paths]
    
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # Setup Model
    model = get_resnet_model().to(device)
    criterion = nn.CrossEntropyLoss()
    # Folosim un learning rate putin mai mic pentru finetuning
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    best_acc = 0.0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            r_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        t_loss = r_loss / len(train_loader)
        t_acc = 100 * correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_loss += criterion(outputs, labels).item()
                _, pred = torch.max(outputs, 1)
                v_total += labels.size(0)
                v_correct += (pred == labels).sum().item()

        v_loss /= len(val_loader)
        v_acc = 100 * v_correct / v_total
        
        print(f"Epoch {epoch+1}: Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}% | Val Loss: {v_loss:.4f}")

        history['epoch'].append(epoch+1)
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        scheduler.step()

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ New Best ResNet Model saved! ({v_acc:.2f}%)")

    # <--- 3. Stop Cronometru
    end_time = time.time()
    
    # <--- 4. Calcul și Afișare
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    time_str = f"Timp total antrenare ResNet18: {minutes}m {seconds}s"
    print("\n" + "="*40)
    print(f"⏱️  {time_str}")
    print("="*40)

    # <--- 5. (Opțional) Salvează într-un fișier text ca să nu uiți
    with open("results/time_resnet.txt", "w") as f:
        f.write(time_str)

    # Salvare CSV
    pd.DataFrame(history).to_csv(HISTORY_SAVE_PATH, index=False)
    print(f"Done. History saved to {HISTORY_SAVE_PATH}")

if __name__ == "__main__":
    main()