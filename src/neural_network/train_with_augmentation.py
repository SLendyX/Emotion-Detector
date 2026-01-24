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

# --- CONFIGURATION ---
BATCH_SIZE = 32
# Lower learning rate is better for fine-tuning pre-trained models
LEARNING_RATE = 0.0001 
EPOCHS = 15
IMAGE_SIZE = 224  # ResNet standard input size
NUM_CLASSES = 7 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
RAF_TRAIN_DIR = 'data/raw/train'
RAF_TEST_DIR = 'data/raw/test'
MY_DATA_DIR = 'data/generated'

# --- 1. BALANCING LOGIC ---
def create_balanced_file_list(raf_dir, my_data_dir):
    """
    Creates a training list where Generated Data is oversampled (repeated)
    to match 40% of the total dataset.
    """
    # Get all file paths (handling jpg and png)
    raf_files = glob.glob(os.path.join(raf_dir, '*', '*.jpg')) + glob.glob(os.path.join(raf_dir, '*', '*.png'))
    gen_files = glob.glob(os.path.join(my_data_dir, '*', '*.jpg')) + glob.glob(os.path.join(my_data_dir, '*', '*.png'))

    num_raf = len(raf_files)
    num_gen = len(gen_files)

    if num_raf == 0:
        raise ValueError(f"No images found in {raf_dir}")
    if num_gen == 0:
        raise ValueError(f"No images found in {my_data_dir}")

    print(f"--- Data Discovery ---")
    print(f"RAF-DB Images: {num_raf}")
    print(f"Your Generated Images: {num_gen}")

    # Calculate target numbers to achieve 40% split
    # RAF (60%) = num_raf -> Total = num_raf / 0.60
    target_total = num_raf / 0.60
    target_gen_count = int(target_total * 0.40)
    
    # Calculate repeat factor
    repeat_factor = math.ceil(target_gen_count / num_gen)
    
    print(f"--- Balancing Logic ---")
    print(f"Target 'Own Data' count: {target_gen_count}")
    print(f"Repeating your data {repeat_factor} times to fill the gap.")

    # Create the oversampled list
    oversampled_gen_files = gen_files * repeat_factor
    oversampled_gen_files = oversampled_gen_files[:target_gen_count]
    
    # Combine and Shuffle
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
        
        # Get label from parent folder name
        parent_folder = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[parent_folder]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image in worst case to prevent crash
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 3. TRANSFORMS (Updated for ResNet 224x224) ---
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
    
    # --- SANITY CHECK ---
    print("\n--- CHECKING FOLDER STRUCTURE ---")
    raf_folders = sorted([d for d in os.listdir(RAF_TRAIN_DIR) if os.path.isdir(os.path.join(RAF_TRAIN_DIR, d))])
    gen_folders = sorted([d for d in os.listdir(MY_DATA_DIR) if os.path.isdir(os.path.join(MY_DATA_DIR, d))])
    
    print(f"RAF Classes: {raf_folders}")
    print(f"Gen Classes: {gen_folders}")
    
    if raf_folders != gen_folders:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("CRITICAL WARNING: Your generated folders do not match RAF-DB folders exactly!")
        print("This will cause label mismatching. Please rename your folders to match.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Proceeding anyway, but user should be warned
    
    # Prepare Data List
    try:
        train_files = create_balanced_file_list(RAF_TRAIN_DIR, MY_DATA_DIR)
        test_files = glob.glob(os.path.join(RAF_TEST_DIR, '*', '*.jpg'))
    except ValueError as e:
        print(f"Error: {e}")
        exit()

    # Create Datasets & Loaders
    train_dataset = MixedEmotionDataset(train_files, raf_folders, transform=train_transforms)
    test_dataset = MixedEmotionDataset(test_files, raf_folders, transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- MODEL: RESNET18 ---
    print("\nInitializing ResNet18...")
    # Load pre-trained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace the final fully connected layer
    # ResNet18's fc layer has 512 input features
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Optimized Learning Rate for transfer learning
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print(f"\nStarting training on {DEVICE}...")
    
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
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
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'models/best_emotion_model.pth')
            print("  --> New Best Model Saved!")

    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")