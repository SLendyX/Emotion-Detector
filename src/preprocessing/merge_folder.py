import os
import shutil

# --- CONFIGURATION ---
# The folder containing your new high-quality images (CK+, KDEF, etc.)
SOURCE_DIR = "kdef" 

# The destination folder (Your main FER training data)
DEST_DIR = "data/raw/train"

CATEGORIES = ["angry", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

def merge_folders():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ Error: Source folder '{SOURCE_DIR}' not found.")
        return

    print(f"🚀 Starting Merge: '{SOURCE_DIR}' -> '{DEST_DIR}'...\n")
    
    total_moved = 0

    for cat in CATEGORIES:
        src_path = os.path.join(SOURCE_DIR, cat)
        dest_path = os.path.join(DEST_DIR, cat)

        # Skip if source folder doesn't exist for this emotion
        if not os.path.exists(src_path):
            print(f"   ⚠️  Skipping {cat}: No folder in source.")
            continue

        # Create destination folder if it doesn't exist (just in case)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        # Get all files in the source category folder
        files = os.listdir(src_path)
        count = 0
        
        for filename in files:
            source_file = os.path.join(src_path, filename)
            
            # Skip directories, only move files
            if os.path.isfile(source_file):
                # Construct destination path
                destination_file = os.path.join(dest_path, filename)
                
                # Handle Duplicate Names: Rename if file already exists
                if os.path.exists(destination_file):
                    base, ext = os.path.splitext(filename)
                    # Create a unique name: "image_merged_1.jpg"
                    new_name = f"{base}_merged{ext}"
                    destination_file = os.path.join(dest_path, new_name)

                # MOVE the file
                shutil.move(source_file, destination_file)
                count += 1

        print(f"   ✅ {cat.upper()}: Moved {count} images.")
        total_moved += count

    print("-" * 40)
    print(f"🎉 SUCCESS! Moved {total_moved} images total.")
    print(f"   The '{SOURCE_DIR}' folder should now be empty.")

if __name__ == "__main__":
    confirm = input(f"⚠️  Move files from '{SOURCE_DIR}' to '{DEST_DIR}'? (yes/no): ")
    if confirm.lower() == "yes":
        merge_folders()
    else:
        print("❌ Operation cancelled.")