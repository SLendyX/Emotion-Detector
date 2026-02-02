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

# --- CONFIGURATION ---
DATA = "data"
GENERATED_DIR = "data/generated"
TRAIN_DIR = os.path.join(DATA, "raw/train")
TEST_DIR =  os.path.join(DATA, "raw/test")
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

        # Helper function to walk directories (Avoids code duplication)
        def load_images_from_folder(directory):
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

        assert len(self.image_paths) == len(self.labels), "Images and Labels count don't match"

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

        # Block 1: 3 -> 32 channels. Output size: 50x50
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 32 -> 64 channels. Output size: 25x25
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ) 

        # Block 3: 64 -> 128 channels. Output size: 12x12
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        #added dropout
        # self.drop = nn.Dropout(p=0.5)

        # Classifier
        self.fc = nn.Linear(128 * 12 * 12, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1) # Flatten
        
        # out= self.drop(out)
        
        out = self.fc(out)
        return out

# --- MAIN TRAINING LOOP ---
def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print(TRAIN_DIR)
    
    # 2. Create Two Distinct Datasets
    train_tf, val_tf = get_transforms()

    # The Training Dataset (With Augmentation)
    train_set = EmotionDataset(raf_dir=TRAIN_DIR, gen_dir=GENERATED_DIR, transform=train_tf)
    
    # The Validation Dataset (Clean, Resize only)
    val_set = EmotionDataset(raf_dir=TEST_DIR, gen_dir="", transform=val_tf)

    # 4. Calculate Sampling Weights (The 40% Logic)
    # We need to scan the full dataset to count the real split ratios
    raf_indices = []
    gen_indices = []

    # Helper scan of the whole dataset logic to get global counts
    # This is an estimation to set the global weights
    for idx, path in enumerate(train_set.image_paths):
        if "generated" in path:
            gen_indices.append(idx)
        else:
            raf_indices.append(idx)
            
    n_gen_total = len(gen_indices)
    n_raf_total = len(raf_indices)
    if n_gen_total == 0: n_gen_total = 1
    if n_raf_total == 0: n_raf_total = 1

    weight_per_gen = 0.4 / n_gen_total
    weight_per_raf = 0.6 / n_raf_total

    # Apply weights specifically to the Training Subset
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
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0.00001)

    # 7. Training Loop
    os.makedirs("models", exist_ok=True)
    
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("Starting Training...")

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

            # Calculate Training Accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Calculate average train metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Store them
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- VALIDATION PHASE ---
        model.eval() 
        val_running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels) # We calculate loss here now too!
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average val metrics
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * correct / total

        # Store them
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print detailed stats
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step()
        
        # Optional: Print LR to confirm it's dropping
        current_lr = scheduler.get_last_lr()
        print(f"Epoch {epoch+1} LR: {current_lr}")
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"models/emotion_model_epoch_{epoch+1}.pt")
            

    # --- PLOTTING ---
    # Once the loop finishes, we plot the graphs
    epochs_range = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # plt.show()
    # Save the plot too
    plt.savefig('training_curves.png')

if __name__ == "__main__":
    main()