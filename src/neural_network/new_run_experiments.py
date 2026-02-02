import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
import time
from sklearn.metrics import f1_score # <--- AM ADAUGAT ACEASTA IMPORTARE

# --- CONFIGURATIE GLOBALA ---
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")
RAF_TRAIN_DIR = os.path.join(RAW_DIR, "train")
RAF_TEST_DIR = os.path.join(RAW_DIR, "test")


BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_map = {
    "angry": 0, "disgust": 1, "fear": 2, "happy": 3, 
    "neutral": 4, "sad": 5, "surprised": 6
}

# --- 1. DATASET & TRANSFORMS ---
class EmotionDataset(Dataset):
    def __init__(self, roots, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if isinstance(roots, str):
            roots = [roots]

        for root_dir in roots:
            for root, dirs, files in os.walk(root_dir, topdown=True):
                folder_name = os.path.basename(root)
                if folder_name not in class_map:
                    continue
                for name in files:
                    if name.endswith(('.jpg', '.png', '.jpeg')):    
                        self.image_paths.append(os.path.join(root, name))
                        self.labels.append(class_map[folder_name])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

# --- 2. MODEL FLEXIBIL ---
class SimpleEmotionCNN(nn.Module):
    def __init__(self, num_classes=7, use_dropout=False):
        super(SimpleEmotionCNN, self).__init__()
        self.use_dropout = use_dropout

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2)
        ) 
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        if self.use_dropout:
            self.drop = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(128 * 12 * 12, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        
        if self.use_dropout:
            out = self.drop(out)
            
        out = self.fc(out)
        return out

# --- 3. EVALUARE (Cu F1 Score) ---
def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Pentru acuratete simpla
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Pentru F1 Score (colectam tot)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = 100 * correct / total
    
    # Calculam F1 Score (Weighted = tine cont de cate exemple sunt in fiecare clasa)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return acc, f1

# --- 4. ANTRENARE ---
def run_experiment(config, train_loader, test_loader):
    print(f"\n>>> Running: {config['name']}")
    print(f"    Params: Dropout={config['dropout']}, WD={config['wd']}, SchedStep={config['step_size']}")
    
    model = SimpleEmotionCNN(num_classes=7, use_dropout=config['dropout']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=config['wd'])
    
    scheduler = None
    if config['step_size'] is not None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    best_test_acc = 0.0
    best_test_f1 = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if scheduler:
            scheduler.step()
            
        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            # Aici primim ambele valori acum
            test_acc, test_f1 = evaluate_on_test(model, test_loader)
            train_loss = running_loss / len(train_loader)
            
            print(f"    Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {train_loss:.4f} | ACC: {test_acc:.2f}% | F1: {test_f1:.4f}")
            
            # Salvam daca avem cea mai buna acuratete (poti schimba sa salvezi dupa F1 daca vrei)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                torch.save(model.state_dict(), f"models/experiments/{config['name']}_best.pt")

    return best_test_acc, best_test_f1

# --- 5. MAIN ---
def main():
    os.makedirs("models", exist_ok=True)
    
    print("Pregatire Dataset...")
    train_tf, val_tf = get_transforms()
    
    # 1. Dataset Train
    dummy_dataset = EmotionDataset([RAF_TRAIN_DIR, GENERATED_DIR], transform=None)
    
    raf_indices = [i for i, p in enumerate(dummy_dataset.image_paths) if "generated" not in p]
    gen_indices = [i for i, p in enumerate(dummy_dataset.image_paths) if "generated" in p]
    
    # Evitam impartirea la 0
    len_gen = len(gen_indices) if len(gen_indices) > 0 else 1
    len_raf = len(raf_indices) if len(raf_indices) > 0 else 1

    weight_per_gen = 0.4 / len_gen
    weight_per_raf = 0.6 / len_raf
    
    weights = [0] * len(dummy_dataset)
    for i in raf_indices: weights[i] = weight_per_raf
    for i in gen_indices: weights[i] = weight_per_gen
    
    train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_dataset = EmotionDataset([RAF_TRAIN_DIR, GENERATED_DIR], transform=train_tf)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    
    # 2. Dataset Test
    test_dataset = EmotionDataset(RAF_TEST_DIR, transform=val_tf)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data Loaded. Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    
    experiments = [
        {"name": "Exp1_Baseline", "dropout": False, "wd": 0.0, "step_size": None, "desc": "Vanilla"},
        {"name": "Exp2_OverfitFighter", "dropout": True, "wd": 1e-4, "step_size": 15, "desc": "Drop+WD+Sched"},
        {"name": "Exp3_Gentle", "dropout": False, "wd": 0.0, "step_size": 20, "desc": "Sched(20) Only"},
        {"name": "Exp4_Aggressive", "dropout": False, "wd": 1e-4, "step_size": 10, "desc": "WD+Sched(10)"}
    ]
    
    results = []

    for config in experiments:
        start_time = time.time()
        # Primim F1 inapoi
        final_acc, final_f1 = run_experiment(config, train_loader, test_loader)
        duration = time.time() - start_time
        
        results.append({
            "Experiment": config["name"],
            "Description": config["desc"],
            "Best Acc (%)": final_acc,
            "Best F1": final_f1, # Adaugat in raport
            "Duration (s)": int(duration)
        })

    print("\n" + "="*60)
    print("             REZULTATE FINALE             ")
    print("="*60)
    df = pd.DataFrame(results)
    df = df.sort_values(by="Best Acc (%)", ascending=False)
    print(df.to_string(index=False))
    
    df.to_csv("docs/experiments/experiment_results_f1.csv", index=False)

if __name__ == "__main__":
    main()