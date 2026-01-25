import os
import glob
import random
import math
import csv  # Added for CSV logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# --- CONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 15
IMAGE_SIZE = 224
NUM_CLASSES = 7 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output Paths
RESULTS_DIR = 'results'
MODEL_SAVE_PATH = os.path.join("models", 'best_emotion_model.pt')
HISTORY_PATH = os.path.join(RESULTS_DIR, 'history.csv')

# Paths
RAF_TRAIN_DIR = 'data/raw/train'
RAF_TEST_DIR = 'data/raw/test'
MY_DATA_DIR = 'data/generated'

# --- 1. BALANCING LOGIC (40% Rule) ---
def create_balanced_file_list(raf_dir, my_data_dir):
    raf_files = glob.glob(os.path.join(raf_dir, '*', '*.jpg')) + glob.glob(os.path.join(raf_dir, '*', '*.png'))
    gen_files = glob.glob(os.path.join(my_data_dir, '*', '*.jpg')) + glob.glob(os.path.join(my_data_dir, '*', '*.png'))

    num_raf = len(raf_files)
    num_gen = len(gen_files)

    if num_raf == 0: raise ValueError(f"No images found in {raf_dir}")
    if num_gen == 0: raise ValueError(f"No images found in {my_data_dir}")

    print(f"--- Data Discovery ---")
    print(f"RAF-DB Images: {num_raf}")
    print(f"Your Generated Images: {num_gen}")

    # Target: Generated should be 40% of TOTAL
    target_total = num_raf / 0.60
    target_gen_count = int(target_total * 0.40)
    
    repeat_factor = math.ceil(target_gen_count / num_gen)
    
    print(f"--- Balancing Logic ---")
    print(f"Target 'Own Data' count: {target_gen_count}")
    print(f"Repeating your data {repeat_factor} times.")

    oversampled_gen_files = gen_files * repeat_factor
    oversampled_gen_files = oversampled_gen_files[:target_gen_count]
    
    final_list = raf_files + oversampled_gen_files
    random.shuffle(final_list)
    
    print(f"Final Training Pool: {len(final_list)} images")
    return final_list

# --- 2. CUSTOM DATASET ---
class MixedEmotionDataset(Dataset):
    def __init__(self, file_list, classes, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        parent_folder = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[parent_folder]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 3. TRANSFORMS ---
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Create results folder
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Prepare CSV File
    with open(HISTORY_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
    print(f"📄 Created history log at: {HISTORY_PATH}")

    # Identify classes
    raf_folders = sorted([d for d in os.listdir(RAF_TRAIN_DIR) if os.path.isdir(os.path.join(RAF_TRAIN_DIR, d))])
    
    # Prepare Data
    try:
        train_files = create_balanced_file_list(RAF_TRAIN_DIR, MY_DATA_DIR)
        test_files = glob.glob(os.path.join(RAF_TEST_DIR, '*', '*.jpg'))
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    train_dataset = MixedEmotionDataset(train_files, raf_folders, transform=train_transforms)
    test_dataset = MixedEmotionDataset(test_files, raf_folders, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- MODEL ---
    print("\nInitializing ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting training on {DEVICE}...")
    best_acc = 0.0

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        test_correct = 0
        test_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels) # Calculate val loss too
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        avg_val_loss = val_running_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {test_acc:.2f}%")
        
        # --- SAVE TO CSV ---
        with open(HISTORY_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, train_acc, avg_val_loss, test_acc])

        # --- SAVE MODEL ---
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> New Best Model Saved!")

    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")
    print(f"History saved to {HISTORY_PATH}")