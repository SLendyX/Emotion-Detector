import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
# Paths to your actual data
RAW_DIR = "data/raw/train"
GEN_DIR = "data/generated"
DOCS_DIR = "docs"

# Categories
CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]

def get_image_paths():
    """
    Scans the directories and returns a dictionary:
    {
        'Angry': ['data/raw/train/angry/1.jpg', 'data/generated/angry/img_1.jpg', ...],
        'Disgust': [...],
        ...
    }
    """
    data_map = {cat: [] for cat in CATEGORIES}
    
    print("📂 Scanning directories...")
    
    # 1. Scan RAF-DB
    for cat in CATEGORIES:
        # Check lowercase and capitalized folder names
        paths = [
            os.path.join(RAW_DIR, cat.lower()), 
            os.path.join(RAW_DIR, cat)
        ]
        
        for p in paths:
            if os.path.exists(p):
                files = glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.png"))
                data_map[cat].extend(files)

    # 2. Scan Generated Data
    for cat in CATEGORIES:
        paths = [
            os.path.join(GEN_DIR, cat.lower()), 
            os.path.join(GEN_DIR, cat)
        ]
        
        for p in paths:
            if os.path.exists(p):
                files = glob.glob(os.path.join(p, "*.jpg")) + glob.glob(os.path.join(p, "*.png"))
                data_map[cat].extend(files)

    total_images = sum(len(v) for v in data_map.values())
    print(f"✅ Found {total_images} images total.")
    return data_map

def main():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # Get all file paths
    data_map = get_image_paths()

    # =========================================================
    # PART A: Class Distribution (Histogram)
    # =========================================================
    print("📊 Generating Class Histogram...")
    
    counts = [len(data_map[cat]) for cat in CATEGORIES]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(CATEGORIES, counts, color='skyblue', edgecolor='black')
    
    # Add counts on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.title('Distribution of Emotion Classes (Raw + Generated)')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    hist_path = os.path.join(DOCS_DIR, "grafice/distributie_clase_direct.png")
    plt.savefig(hist_path)
    print(f"✅ Histogram saved to: {hist_path}")
    plt.close()

    # =========================================================
    # PART B: Random Samples Visualization
    # =========================================================
    print("🖼️ Generating Sample Preview...")

    plt.figure(figsize=(15, 3))

    for i, cat in enumerate(CATEGORIES):
        file_list = data_map[cat]
        
        if not file_list:
            print(f"⚠️ No images found for {cat}")
            continue

        # Pick random image
        random_path = random.choice(file_list)
        
        # Load directly using OpenCV
        img = cv2.imread(random_path)
        # Convert BGR to RGB for Matplotlib
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(1, 7, i + 1)
            plt.imshow(img)
            plt.title(f"{cat}\n({len(file_list)})")
            plt.axis('off')
        else:
            print(f"⚠️ Could not load {random_path}")

    plt.tight_layout()
    samples_path = os.path.join(DOCS_DIR, "grafice/esantioane_emotii_direct.png")
    plt.savefig(samples_path)
    print(f"✅ Samples saved to: {samples_path}")
    plt.close()

    # =========================================================
    # PART C: Pixel Intensity Analysis (Boxplot & Histogram)
    # =========================================================
    print("📉 Generating Pixel Intensity Analysis (Sampling 50 images/class)...")
    
    SAMPLE_SIZE = 50 
    
    pixel_intensities = [] # For histogram (all pixels)
    mean_intensities_per_class = [] # For boxplot (mean per image)
    
    for cat in CATEGORIES:
        file_list = data_map[cat]
        if not file_list:
            mean_intensities_per_class.append([])
            continue
            
        # Sample random files to avoid Out Of Memory errors
        current_sample = random.sample(file_list, min(len(file_list), SAMPLE_SIZE))
        class_means = []
        
        for path in current_sample:
            img = cv2.imread(path)
            if img is not None:
                # Convert BGR to Gray for brightness stats
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # For Boxplot: Mean brightness of this specific image
                class_means.append(np.mean(gray))
                
                # For Histogram: Add a random subset of pixels (to keep it fast)
                # We flatten and take every 100th pixel to approximate distribution
                pixel_intensities.extend(gray.flatten()[::100])
        
        mean_intensities_per_class.append(class_means)

    for i, cat in enumerate(CATEGORIES):
        # Convertim lista de medii (0-255) în array numpy
        raw_values = np.array(mean_intensities_per_class[i])
        
        if len(raw_values) == 0:
            print(f"\n--- {cat}: Fără date ---")
            continue

        # 1. Normalizare la 0-1
        norm_values = raw_values / 255.0

        # 2. Calcul Quartile și IQR
        q1 = np.percentile(norm_values, 25)
        q3 = np.percentile(norm_values, 75)
        iqr = q3 - q1

        # 3. Calcul Limite (metoda standard Tukey: 1.5 * IQR)
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        # 4. Identificare Outlieri
        # Outlieri întunecați: valori strict mai mici decât limita inferioară
        dark_outliers = norm_values[norm_values < lower_limit]
        # Outlieri luminoși: valori strict mai mari decât limita superioară
        bright_outliers = norm_values[norm_values > upper_limit]

        print(f"\nCategoria: {cat}")
        print(f"  * IQR (Interquartile Range): {iqr:.4f}")
        print(f"  * Limita inferioară (Prea întunecat): {lower_limit:.4f}")
        print(f"  * Limita superioară (Prea luminos): {upper_limit:.4f}")
        print(f"  * Imagini outlier întunecate: {len(dark_outliers)}")
        print(f"  * Imagini outlier luminoase: {len(bright_outliers)}")

    # 1. Boxplot (Brightness per Class)
    plt.figure(figsize=(10, 6))
    plt.boxplot(mean_intensities_per_class, labels=CATEGORIES, patch_artist=True)
    plt.title('Distribution of Image Brightness per Class\n(Are some emotions darker?)')
    plt.ylabel('Mean Pixel Intensity (0=Black, 255=White)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    box_path = os.path.join(DOCS_DIR, "grafice/boxplot_intensitate.png")
    plt.savefig(box_path)
    print(f"✅ Boxplot saved to: {box_path}")
    plt.close()

    # 2. Pixel Histogram (Global)
    plt.figure(figsize=(10, 6))
    plt.hist(pixel_intensities, bins=50, color='gray', alpha=0.7, edgecolor='black')
    plt.title('Global Pixel Intensity Distribution (Sampled)')
    plt.xlabel('Pixel Value (0=Black, 255=White)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    mean_val = np.mean(pixel_intensities)
    median_val = np.median(pixel_intensities)
    std_val = np.std(pixel_intensities)

    print(f"Medie: {mean_val/255.0:.2f}")
    print(f"Mediană: {median_val/255.0:.2f}")
    print(f"Deviație standard: {std_val/255.0:.2f}")
    
    pixels_np = np.array(pixel_intensities)

    # Calculate raw statistics (0-255 range)
    min_raw = np.min(pixels_np)
    max_raw = np.max(pixels_np)
    q1_raw = np.percentile(pixels_np, 25)
    q3_raw = np.percentile(pixels_np, 75)

    print("\n--- Statistici Detaliate (Normalizate 0-1) ---")
    # We divide by 255.0 to get the 0.0 - 1.0 range you asked for
    print(f"* Min: {min_raw:.1f}, Max: {max_raw:.1f}")
    print(f"* Q1 (25%): {q1_raw:.4f}")
    print(f"* Q3 (75%): {q3_raw:.4f}")

    # Opțional: Adăugăm linii verticale pe grafic pentru a le vizualiza
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Medie: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='dotted', linewidth=2, label=f'Mediană: {median_val:.1f}')
    plt.legend()
    
    pixel_hist_path = os.path.join(DOCS_DIR, "grafice/histograma_pixeli.png")
    plt.savefig(pixel_hist_path)
    print(f"✅ Pixel Histogram saved to: {pixel_hist_path}")
    plt.close()

if __name__ == "__main__":
    main()