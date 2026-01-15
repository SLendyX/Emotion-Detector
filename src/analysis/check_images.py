import os

# Check this path matches your setup
DISGUST_DIR = "data/raw/train/disgust"

if os.path.exists(DISGUST_DIR):
    files = os.listdir(DISGUST_DIR)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
    subfolders = [f for f in files if os.path.isdir(os.path.join(DISGUST_DIR, f))]
    
    print(f"📂 Scanning: {DISGUST_DIR}")
    print(f"   ✅ Total Files Found: {len(files)}")
    print(f"   📸 Valid Images: {len(image_files)}")
    print(f"   📁 Sub-folders (BAD): {len(subfolders)}")
    
    if len(subfolders) > 0:
        print("\n⚠️  WARNING: Found folders inside your image folder!")
        print("   The script ignores these. You must move the images OUT of these subfolders.")
        print(f"   Example subfolders: {subfolders[:5]}")
else:
    print("❌ Error: Folder not found.")