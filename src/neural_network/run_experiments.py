import os
import glob
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score
import numpy as np

# --- CONFIGURARE GENERALĂ ---
BATCH_SIZE = 32
EPOCHS = 15
IMAGE_SIZE = 224
NUM_CLASSES = 7 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = 'docs/optimization'
MODELS_DIR = 'models'

# Căi date
RAF_TRAIN_DIR = 'data/raw/train'
RAF_TEST_DIR = 'data/raw/test'
MY_DATA_DIR = 'data/generated'

# Asigurare directoare
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 1. DATASET & BALANCING ---
def create_balanced_file_list(raf_dir, my_data_dir):
    raf_files = glob.glob(os.path.join(raf_dir, '*', '*.jpg')) + glob.glob(os.path.join(raf_dir, '*', '*.png'))
    gen_files = glob.glob(os.path.join(my_data_dir, '*', '*.jpg')) + glob.glob(os.path.join(my_data_dir, '*', '*.png'))
    
    target_total = len(raf_files) / 0.60
    target_gen_count = int(target_total * 0.40)
    repeat_factor = math.ceil(target_gen_count / max(1, len(gen_files)))
    
    oversampled_gen = (gen_files * repeat_factor)[:target_gen_count]
    final_list = raf_files + oversampled_gen
    random.shuffle(final_list)
    return final_list

class EmotionDataset(Dataset):
    def __init__(self, file_list, classes, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
    
    def __len__(self): return len(self.file_list)
    
    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.class_to_idx[os.path.basename(os.path.dirname(path))]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform: img = self.transform(img)
        return img, label

# --- 2. CONFIGURARE EXPERIMENTE ---
def get_experiment_config(exp_id):
    config = {
        'lr': 0.0003,
        'architecture': 'resnet18', 
        'dropout': 0.5,
        'weight_decay': 0.0,
        'name': f'Exp_{exp_id}'
    }
    
    if exp_id == 1:
        config['name'] = 'Exp_1_Baseline'
    elif exp_id == 2:
        config['name'] = 'Exp_2_HighLR'
        config['lr'] = 0.001
    elif exp_id == 3:
        config['name'] = 'Exp_3_DeepArchitecture'
        config['architecture'] = 'resnet18_deep'
        config['dropout'] = 0.55
    elif exp_id == 4:
        config['name'] = 'Exp_4_HighReg'
        config['dropout'] = 0.5
        config['weight_decay'] = 0.01
        
    return config

def build_model(config):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    
    if config['architecture'] == 'resnet18_deep':
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(1024, NUM_CLASSES)
        )
    else:
        model.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(num_ftrs, NUM_CLASSES)
        )
    return model.to(DEVICE)

# --- 3. FUNCȚIE DE ANTRENARE ---
def run_training(exp_id):
    cfg = get_experiment_config(exp_id)
    print(f"\n🚀 Starting {cfg['name']} | LR: {cfg['lr']} | Arch: {cfg['architecture']} | L2: {cfg['weight_decay']}")
    
    # Date
    classes = sorted([d for d in os.listdir(RAF_TRAIN_DIR) if os.path.isdir(os.path.join(RAF_TRAIN_DIR, d))])
    train_files = create_balanced_file_list(RAF_TRAIN_DIR, MY_DATA_DIR)
    test_files = glob.glob(os.path.join(RAF_TEST_DIR, '*', '*.jpg'))
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(EmotionDataset(train_files, classes, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(EmotionDataset(test_files, classes, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = build_model(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    best_acc = 0.0
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
        # Validate (Full Pass for F1 Score)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate Metrics
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"   Ep {epoch+1}: Acc {val_acc*100:.2f}% | F1 {val_f1:.4f}")
        
        # Save best by Accuracy (but tracking F1 too)
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{cfg['name']}.pt"))

    return best_acc, best_f1

def main():
    results = []
    best_overall_acc = 0
    best_exp_config = None

    for i in range(1, 5):
        acc, f1 = run_training(i)
        results.append({'Exp': f"Exp {i}", 'Acc': acc, 'F1': f1})
        
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_exp_config = get_experiment_config(i)
            
            # Save as optimized immediately
            src_path = os.path.join(MODELS_DIR, f"{best_exp_config['name']}.pt")
            dst_path = os.path.join(MODELS_DIR, "optimized_model.pt")
            
            # Re-load and save to ensure consistency
            model = build_model(best_exp_config)
            model.load_state_dict(torch.load(src_path))
            torch.save(model.state_dict(), dst_path)

    print("\n=== REZULTATE FINALE PENTRU TABEL ===")
    print(f"{'Exp':<10} | {'Acc':<10} | {'F1-Score':<10}")
    print("-" * 35)
    for res in results:
        print(f"{res['Exp']:<10} | {res['Acc']*100:.2f}%     | {res['F1']:.4f}")
    
    print(f"\nCâștigător salvat ca 'optimized_model.pt': {best_exp_config['name']}")

if __name__ == "__main__":
    main()