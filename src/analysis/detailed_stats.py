import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURARE ---
DATA_DIR = "data/processed"
DOCS_DIR = "docs"

def calculate_and_plot():
    print("🔄 Încărcare date...")
    try:
        # Încărcăm doar setul de antrenare pentru analiză
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        # X_train are forma (Nr_imagini, 48, 48, 1)
    except FileNotFoundError:
        print("❌ Eroare: Nu găsesc X_train.npy. Rulează întâi preprocesarea!")
        return

    # 1. Pregătirea datelor pentru statistică
    # "Aplatizăm" datele pentru a avea un șir lung de pixeli (pentru statistici globale)
    all_pixels = X_train.flatten()
    
    # Calculăm media intensității PENTRU FIECARE IMAGINE în parte (pentru outlieri)
    # axis=(1, 2, 3) face media pe înălțime, lățime și canale
    image_means = np.mean(X_train, axis=(1, 2, 3))

    print("\n--- 📊 REZULTATE PENTRU README (Capitolul 3) ---")
    
    # --- A. MEDIE, MEDIANĂ, DEVIAȚIE STANDARD ---
    mean_val = np.mean(all_pixels)
    median_val = np.median(all_pixels)
    std_val = np.std(all_pixels)
    
    print(f"\n1. Statistici Globale (Pixeli):")
    print(f"   * Medie: {mean_val:.4f} (Ideal ~0.5 pentru date normalizate)")
    print(f"   * Mediană: {median_val:.4f}")
    print(f"   * Deviație Standard: {std_val:.4f}")

    # --- B. MIN-MAX și QUARTILE ---
    min_val = np.min(all_pixels)
    max_val = np.max(all_pixels)
    q1 = np.percentile(all_pixels, 25)
    q3 = np.percentile(all_pixels, 75)

    print(f"\n2. Min-Max și Quartile:")
    print(f"   * Min: {min_val:.1f}, Max: {max_val:.1f}")
    print(f"   * Q1 (25%): {q1:.4f}")
    print(f"   * Q3 (75%): {q3:.4f}")

    # --- C. IDENTIFICAREA OUTLIERILOR (IQR pe Intensitatea Medie a Imaginilor) ---
    # Căutăm imagini care sunt global prea întunecate sau prea luminoase
    img_q1 = np.percentile(image_means, 25)
    img_q3 = np.percentile(image_means, 75)
    iqr = img_q3 - img_q1
    
    lower_bound = img_q1 - 1.5 * iqr
    upper_bound = img_q3 + 1.5 * iqr
    
    outliers_dark = np.sum(image_means < lower_bound)
    outliers_bright = np.sum(image_means > upper_bound)
    
    print(f"\n3. Analiza Outlierilor (bazat pe luminozitatea medie a imaginilor):")
    print(f"   * IQR (Interquartile Range): {iqr:.4f}")
    print(f"   * Limita inferioară (Prea întunecat): {lower_bound:.4f}")
    print(f"   * Limita superioară (Prea luminos): {upper_bound:.4f}")
    print(f"   * Imagini outlier întunecate: {outliers_dark}")
    print(f"   * Imagini outlier luminoase: {outliers_bright}")

    # --- GENERARE GRAFICE ---
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # Grafic 1: Histograma Distribuției Pixelilor
    plt.figure(figsize=(10, 5))
    plt.hist(all_pixels[::100], bins=50, color='gray', alpha=0.7) # Luăm un eșantion (1 la 100) pt viteză
    plt.title('Distribuția Intensității Pixelilor (Global)')
    plt.xlabel('Intensitate Pixel (0.0 - 1.0)')
    plt.ylabel('Număr Pixeli (eșantion)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(DOCS_DIR, "histograma_pixeli.png"))
    plt.close()

    # Grafic 2: Boxplot pentru Luminozitatea Imaginilor (Outlieri)
    plt.figure(figsize=(8, 5))
    plt.boxplot(image_means, vert=False)
    plt.title('Boxplot: Luminozitatea Medie a Imaginilor')
    plt.xlabel('Luminozitate Medie (0=Negru, 1=Alb)')
    plt.yticks([])
    plt.grid(True, axis='x', alpha=0.3)
    # Desenăm liniile de outlier
    plt.axvline(lower_bound, color='r', linestyle='--', label='Limită Outlier')
    plt.axvline(upper_bound, color='r', linestyle='--', label='Limită Outlier')
    plt.legend()
    plt.savefig(os.path.join(DOCS_DIR, "boxplot_outlieri.png"))
    plt.close()

    print(f"\n✅ Grafice salvate în '{DOCS_DIR}': histograma_pixeli.png, boxplot_outlieri.png")

if __name__ == "__main__":
    calculate_and_plot()