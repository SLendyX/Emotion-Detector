import cv2
import os
import re
import numpy as np

def crop_faces_by_convention(source_dir, dest_dir, convention_pattern=r"^\d+_\d+\.(jpg|png|jpeg)$"):
    """
    Scans source_dir for images matching the naming convention,
    detects faces, crops them, and saves them to dest_dir.
    
    Args:
        source_dir (str): Path to folder containing raw images (can contain subfolders).
        dest_dir (str): Path where cropped face images will be saved (structure replicated).
        convention_pattern (str): Regex pattern for valid filenames.
    """
    
    # Load Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("❌ Error: Could not load Haar Cascade.")
        return

    pattern = re.compile(convention_pattern, re.IGNORECASE)
    processed_count = 0
    
    print(f"🔍 Scanning {source_dir} for files matching {convention_pattern}...")

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 1. Check Naming Convention
            if pattern.match(file):
                input_path = os.path.join(root, file)
                
                # Determine relative path for output structure
                rel_path = os.path.relpath(root, source_dir)
                output_folder = os.path.join(dest_dir, rel_path)
                
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                    
                # 2. Load Image
                img = cv2.imread(input_path)
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 3. Detect Face
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # If face found, crop the largest one
                if len(faces) > 0:
                    # Pick largest face (w * h)
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    (x, y, w, h) = faces[0]
                    
                    # Crop
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Resize to 48x48 (Standard for this project)
                    face_roi = cv2.resize(face_roi, (48, 48))
                    
                    # Save
                    output_path = os.path.join(output_folder, file)
                    cv2.imwrite(output_path, face_roi)
                    processed_count += 1
                    if processed_count % 50 == 0:
                        print(f"   Processed {processed_count} images...")
                else:
                    print(f"⚠️ No face detected in {file}")

    print(f"\n✅ Finished! Processed {processed_count} images.")
    print(f"📂 Saved to: {dest_dir}")

if __name__ == "__main__":
    # Example Usage
    # Assuming 'kdef' has raw photos and we want to output to 'data/generated' or 'data/raw/train'
    # You can adjust these paths
    SOURCE_ROOT = "data/raw/train" 
    DEST_ROOT = "data/kdef_cropped" # Intermediate folder before merging
    
    crop_faces_by_convention(SOURCE_ROOT, DEST_ROOT)
