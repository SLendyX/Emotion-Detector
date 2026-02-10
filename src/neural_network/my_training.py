import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
import torchvision
from torchvision import transforms, models
from PIL import Image
import time

# --- CONFIGURATION ---
DATA = "data"
GENERATED_DIR = "data/generated"
TRAIN_DIR = os.path.join(DATA, "raw/train")
TEST_DIR =  os.path.join(DATA, "raw/test")
RESULTS_DIR = "results"  # Folder for CSV results
DOCS_DIR = "docs/grafice" # Folder for plots

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

class_map = {
    "angry": 0, 
    "disgust": 1, 
    "fear": 2, 
    "happy": 3, 
    "neutral": 4, 
    "sad": 5, 
    "surprised": 6
}

# --- DATASET CLASS ---
class EmotionDataset(Dataset):
    def __init__(self, raf_dir, gen_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Helper function to walk directories
        def load_images_from_folder(directory):
            if not os.path.exists(directory):
                return
            for root, dirs, files in os.walk(directory, topdown=True):
                folder_name = os.path.basename(root)
                if folder_name not in class_map:
                    continue
                for name in files:
                    if name.endswith(('.jpg', '.png', '.jpeg')):    
                        full_image_path = os.path.join(root, name)
                        self.image_paths.append(full_image_path)
                        self.labels.append(class_map.get(folder_name))

        # Load both sources
        load_images_from_folder(raf_dir)
        load_images_from_folder(gen_dir)

        # assert len(self.image_paths) > 0, "No images found! Check paths."
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
            
        # Use PIL to load
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# --- TRANSFORMS ---
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms

# --- MODEL ARCHITECTURE ---
class SimpleEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleEmotionCNN, self).__init__()

        # Block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ) 

        # Block 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        # Classifier
        self.fc = nn.Linear(128 * 12 * 12, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out

# --- MAIN TRAINING LOOP ---
def main():
    # 1. Setup Directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/latest_checkpoints", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 3. Create Datasets
    train_tf, val_tf = get_transforms()
    train_set = EmotionDataset(raf_dir=TRAIN_DIR, gen_dir=GENERATED_DIR, transform=train_tf)
    val_set = EmotionDataset(raf_dir=TEST_DIR, gen_dir="", transform=val_tf)

    # 4. Calculate Sampling Weights
    raf_indices = []
    gen_indices = []

    for idx, path in enumerate(train_set.image_paths):
        if "generated" in path:
            gen_indices.append(idx)
        else:
            raf_indices.append(idx)
            
    n_gen_total = len(gen_indices) if len(gen_indices) > 0 else 1
    n_raf_total = len(raf_indices) if len(raf_indices) > 0 else 1

    weight_per_gen = 0.4 / n_gen_total
    weight_per_raf = 0.6 / n_raf_total

    train_weights = []
    for img_path in train_set.image_paths:
        if "generated" in img_path:
            train_weights.append(weight_per_gen)
        else:
            train_weights.append(weight_per_raf)

    # 5. Loaders
    train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 6. Model Setup
    model = SimpleEmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    patience = 15           
    patience_counter = 0   
    best_val_loss = float('inf') 

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }
    
    print("Starting Training...")

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # --- VALIDATION PHASE ---
        model.eval() 
        val_running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]

        # Store History
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # Check Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"✅ Model saved (Best Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter}/{patience} epochs.")
            
            if patience_counter >= patience:
                print(f"🛑 EARLY STOPPING TRIGGERED at epoch {epoch+1}!")
                break 

        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/latest_checkpoints/emotion_model_epoch_{epoch+1}.pt")
           
           
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
            
    # --- SAVE HISTORY TO CSV ---
    print("💾 Saving training history...")
    df = pd.DataFrame(history)
    csv_path = os.path.join(RESULTS_DIR, "training_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Training history saved to: {csv_path}")

    # --- PLOTTING ---
    # Use actual number of epochs run (in case of early stopping)
    actual_epochs = history['epoch']

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(actual_epochs, history['train_acc'], label='Training Accuracy')
    plt.plot(actual_epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(actual_epochs, history['train_loss'], label='Training Loss')
    plt.plot(actual_epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(DOCS_DIR, 'training_curves.png')
    plt.savefig(plot_path)
    print(f"✅ Training curves saved to: {plot_path}")

if __name__ == "__main__":
    main()