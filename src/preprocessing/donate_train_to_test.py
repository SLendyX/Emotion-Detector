import os
import shutil
import random

# --- CONFIGURATION ---
BASE_DIR = "data/raw"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# MINIMUM images required in Train before we allow donation
# We want to keep at least 1000 real images for training if possible.
MIN_TRAIN_THRESHOLD = 1000 

# Targets to move if we have enough
TARGETS_TO_DONATE = {
    "angry": 50,
    "disgust": 100, 
    "fear": 100,
    "happy": 0,    # Happy is already huge in test, no need to add
    "neutral": 0,  # Neutral is usually fine
    "sadness": 50,
    "surprise": 50
}

def safe_donate():
    print("📦 SAFE DONATION CHECK...\n")
    
    for cat, amount in TARGETS_TO_DONATE.items():
        src = os.path.join(TRAIN_DIR, cat)
        dst = os.path.join(TEST_DIR, cat)
        
        if not os.path.exists(src): continue
        if not os.path.exists(dst): os.makedirs(dst)
        
        files = os.listdir(src)
        count = len(files)
        
        # --- THE SAFETY CHECK ---
        # If moving the files would drop us below the safety line, DON'T DO IT.
        if (count - amount) < MIN_TRAIN_THRESHOLD:
            print(f"   ⚠️  Skipping {cat.upper()}: Has {count} images. (Too poor to donate)")
            continue
            
        # Randomly select files to move
        random.shuffle(files)
        to_move = files[:amount]
        
        for f in to_move:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
            
        print(f"   ✅ {cat.upper()}: Donated {len(to_move)} images to Test.")

    print("\nDONE.")

if __name__ == "__main__":
    safe_donate()