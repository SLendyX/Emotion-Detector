import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMAGE_SIZE = 100
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
RAF_TEST_DIR = 'data/raw/test'
MODEL_PATH = 'models/latest_checkpoints/emotion_model_epoch_50.pt'
RESULTS_DIR = 'results'  # <--- Folder pentru rezultate
METRICS_FILE = os.path.join(RESULTS_DIR, 'test_metrics.json') # <--- Fisier destinatie

# --- 1. UTILS & DATASET ---
class SimpleEmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        # Detect classes just like training script
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

# --- 2. TRANSFORMS ---
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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


# --- 3. MAIN EVALUATION ---
if __name__ == "__main__":
    print(f"Running evaluation on {DEVICE}...")

    # Load Dataset
    test_dataset = SimpleEmotionDataset(RAF_TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Test Set Size: {len(test_dataset)} images")
    print(f"Classes: {test_dataset.classes}")

    # Load Model
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

    all_preds = []
    all_labels = []

    # Inference Loop
    print("Starting inference...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # --- METRICS & REPORT ---
    print("\n" + "="*30)
    print("       EVALUATION REPORT       ")
    print("="*30)
    
    # 1. Print Text Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    # 2. Get Dictionary for JSON
    report_dict = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
    
    # Extract Macro Metrics
    metrics_json = {
        "accuracy": report_dict['accuracy'],
        "macro_precision": report_dict['macro avg']['precision'],
        "macro_recall": report_dict['macro avg']['recall'],
        "macro_f1": report_dict['macro avg']['f1-score']
    }

    # 3. Save to JSON
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_json, f, indent=4)
        
    print(f"✅ Metrics saved to JSON: {METRICS_FILE}")

    # 4. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_dataset.classes, 
                yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save the plot
    docs_dir = 'docs/grafice'
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        
    save_path = os.path.join(docs_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix saved to {save_path}")