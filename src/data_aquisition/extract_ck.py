import pandas as pd
import numpy as np
import cv2
import os

# --- CONFIGURATION ---
CSV_PATH = "ckextended.csv"  # The file you downloaded
OUTPUT_DIR = "kdef" # Where to save the new images

"""
0 : Anger (45 samples)
1 : Disgust (59 samples)
2 : Fear (25 samples)
3 : Happiness (69 samples)
4 : Sadness (28 samples)
5 : Surprise (83 samples)
6 : Neutral (593 samples)
7 : Contempt (18 samples)
"""
TARGET_EMOTIONS = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral"
}

def extract_from_csv():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Could not find {CSV_PATH}")
        return

    print(f"📖 Reading {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Check for likely column names
    # Common names: 'pixels', 'emotion', 'Usage', 'image'
    cols = [c.lower() for c in df.columns]
    
    pixel_col = None
    emotion_col = None
    
    # 1. Detect Pixel Column
    for c in df.columns:
        if 'pixel' in c.lower() or 'image' in c.lower():
            pixel_col = c
            break
            
    # 2. Detect Emotion Column
    for c in df.columns:
        if 'emotion' in c.lower() or 'label' in c.lower():
            emotion_col = c
            break

    if not pixel_col or not emotion_col:
        print("❌ Could not auto-detect 'pixels' or 'emotion' columns.")
        print(f"   Found columns: {df.columns.tolist()}")
        print("   Please rename columns in CSV or update the script.")
        return

    print(f"✅ Found Pixel Column: '{pixel_col}'")
    print(f"✅ Found Emotion Column: '{emotion_col}'")

    count = 0
    
    for index, row in df.iterrows():
        emotion_idx = int(row[emotion_col])
        
        # Only process if it's Disgust or Fear
        if emotion_idx in TARGET_EMOTIONS:
            label_name = TARGET_EMOTIONS[emotion_idx]
            
            # Create folder if not exists
            target_folder = os.path.join(OUTPUT_DIR, label_name)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # Convert Pixel String to Image
            # Pixels are usually space-separated strings: "23 45 120 ..."
            try:
                pixels = np.fromstring(str(row[pixel_col]), sep=' ')
                
                # Check if we need to reshape (48x48 is standard)
                if len(pixels) == 48*48:
                    image = pixels.reshape(48, 48)
                else:
                    # Try to guess shape (square root)
                    side = int(np.sqrt(len(pixels)))
                    image = pixels.reshape(side, side)

                image = image.astype(np.uint8)
                
                # Resize to 48x48 just in case it's different
                if image.shape != (48, 48):
                    image = cv2.resize(image, (48, 48))

                # Save Image
                filename = f"ck_extended_{index}.jpg"
                save_path = os.path.join(target_folder, filename)
                cv2.imwrite(save_path, image)
                count += 1
                
            except Exception as e:
                print(f"⚠️ Error parsing row {index}: {e}")
                continue

    print(f"\n🎉 Success! Extracted {count} images to {OUTPUT_DIR}")
    print("   Now run your preprocessing script again to include these new photos.")

if __name__ == "__main__":
    extract_from_csv()