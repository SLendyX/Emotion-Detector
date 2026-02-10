import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import time

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Antrenare Experimentala Emotion CNN')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--batch', type=int, default=32, help='Batch Size')
parser.add_argument('--epochs', type=int, default=50, help='Numar Epoci')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout Rate (0.0 = fara dropout)')
parser.add_argument('--name', type=str, required=True, help='Numele experimentului (ex: exp1)')
args = parser.parse_args()

# --- CONFIGURATION ---
DATA = "data"
GENERATED_DIR = "data/generated"
TRAIN_DIR = os.path.join(DATA, "raw/train")
TEST_DIR =  os.path.join(DATA, "raw/test")
RESULTS_DIR = "results/experiments" # Folder separat pentru experimente
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models/experiments", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Experiment: {args.name} | LR: {args.lr} | Batch: {args.batch} | Dropout: {args.dropout} | Device: {DEVICE}")

class_map = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5, "surprised": 6}

# --- DATASET ---
class EmotionDataset(Dataset):
    def __init__(self, raf_dir, gen_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        def load(directory):
            if not os.path.exists(directory): return
            for root, _, files in os.walk(directory):
                folder = os.path.basename(root)
                if folder in class_map:
                    for f in files:
                        if f.endswith(('.jpg', '.png')):
                            self.image_paths.append(os.path.join(root, f))
                            self.labels.append(class_map[folder])
        load(raf_dir)
        load(gen_dir)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

# --- MODEL (MODIFICAT PENTRU DROPOUT) ---
class SimpleEmotionCNN(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.0):
        super(SimpleEmotionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        
        # Dropout variabil
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(128 * 12 * 12, num_classes)

    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(x)))
        out = out.view(out.size(0), -1)
        if self.drop.p > 0:
            out = self.drop(out)
        out = self.fc(out)
        return out

# --- MAIN ---
def main():
    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Data Loaders
    train_set = EmotionDataset(TRAIN_DIR, GENERATED_DIR, transform=train_tf)
    val_set = EmotionDataset(TEST_DIR, "", transform=val_tf)

    # Weights
    raf_idxs = [i for i, p in enumerate(train_set.image_paths) if "generated" not in p]
    gen_idxs = [i for i, p in enumerate(train_set.image_paths) if "generated" in p]
    w_gen = 0.4 / (len(gen_idxs) or 1)
    w_raf = 0.6 / (len(raf_idxs) or 1)
    weights = [w_gen if "generated" in p else w_raf for p in train_set.image_paths]
    
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)

    # Model Setup
    model = SimpleEmotionCNN(num_classes=7, dropout_rate=args.dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        r_loss, correct, total = 0.0, 0, 0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            
            r_loss += loss.item()
            _, pred = torch.max(out, 1)
            total += lbls.size(0)
            correct += (pred == lbls).sum().item()

        t_loss = r_loss / len(train_loader)
        t_acc = 100 * correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs)
                v_loss += criterion(out, lbls).item()
                _, pred = torch.max(out, 1)
                v_total += lbls.size(0)
                v_correct += (pred == lbls).sum().item()

        v_loss /= len(val_loader)
        v_acc = 100 * v_correct / v_total
        scheduler.step()

        print(f"E[{epoch+1}/{args.epochs}] T_Acc: {t_acc:.1f}% | V_Acc: {v_acc:.1f}% | V_Loss: {v_loss:.3f}")

        history['epoch'].append(epoch+1)
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        # Save Best Model
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), f"models/experiments/{args.name}_best.pt")

    # Save Results
    pd.DataFrame(history).to_csv(f"{RESULTS_DIR}/{args.name}_history.csv", index=False)
    
    duration = time.time() - start_time
    print(f"✅ {args.name} Finished in {int(duration//60)}m {int(duration%60)}s. Best Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()