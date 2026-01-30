import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import glob
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMAGE_SIZE = 100
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
RAF_TRAIN_DIR = 'data/raw/train' # Needed just to get class names
RAF_TEST_DIR = 'data/raw/test'
MODEL_PATH = 'models/emotion_model_epoch_50.pt'

# --- 1. UTILS & DATASET (Same as training) ---
class SimpleEmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        # Detect classes just like training script to ensure index consistency
        self.classes = sorted([d for d in os.listdir(RAF_TEST_DIR) if os.path.isdir(os.path.join(RAF_TEST_DIR, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Get all image files in test dir
        self.file_list = glob.glob(os.path.join(root_dir, '*', '*.jpg')) + glob.glob(os.path.join(root_dir, '*', '*.png'))
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        parent_folder = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[parent_folder]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. TRANSFORMS (Must match training validation transforms) ---
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

        # Classifier
        self.fc = nn.Linear(128 * 12 * 12, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out


# --- 3. MAIN EVALUATION ---
if __name__ == "__main__":
    print(f"Running evaluation on {DEVICE}...")

    # Load Dataset
    test_dataset = SimpleEmotionDataset(RAF_TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Test Set Size: {len(test_dataset)} images")
    print(f"Classes: {test_dataset.classes}")

    # Load Model Structure (Resnet18)
    # model = models.resnet18(weights=None) # No need to download weights, we are loading ours
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    
    # Load Model structure custom
    model = SimpleEmotionCNN(num_classes=7)
    
    # Load Trained Weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Could not find {MODEL_PATH}")
        exit()

    model = model.to(DEVICE)
    model.eval()

    # Lists to store results
    all_preds = []
    all_labels = []

    # Inference Loop
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- METRICS ---
    print("\n" + "="*30)
    print("       EVALUATION REPORT       ")
    print("="*30)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_dataset.classes, 
                yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot
    save_path = 'docs/confusion_matrix.png'
    plt.savefig(save_path)
    print(f"\nConfusion Matrix saved to {save_path}")
    print("You can view the image to see which emotions are being confused.")