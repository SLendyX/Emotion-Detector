import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURARE ---
RAW_DIR = "data/raw/train"   # Folderul cu imagini reale
GEN_DIR = "data/generated"   # Folderul cu imagini generate
DOCS_DIR = "docs"            # Folderul unde salvăm graficul
OUTPUT_FILE = os.path.join(DOCS_DIR, "generated_vs_real.png")

# Categoriile de emoții
CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]

def count_images_in_dir(base_dir):
    """
    Numără imaginile din fiecare categorie dintr-un director dat.
    Caută atât foldere cu nume gen 'Angry' cât și 'angry'.
    """
    counts = []
    print(f"📂 Scanare director: {base_dir}...")
    
    for cat in CATEGORIES:
        # Verificăm ambele variante de nume (Majusculă și litere mici)
        possible_paths = [
            os.path.join(base_dir, cat),
            os.path.join(base_dir, cat.lower())
        ]
        
        count = 0
        found_path = False
        
        for p in possible_paths:
            if os.path.exists(p):
                # Numărăm jpg, png, jpeg
                files = glob.glob(os.path.join(p, "*.jpg")) + \
                        glob.glob(os.path.join(p, "*.png")) + \
                        glob.glob(os.path.join(p, "*.jpeg"))
                count += len(files)
                found_path = True
        
        counts.append(count)
        if not found_path:
            print(f"   ⚠️ Atenție: Nu am găsit folder pentru '{cat}' în {base_dir}")
            
    return counts

def main():
    # 1. Asigurăm existența folderului docs
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # 2. Colectăm datele
    real_counts = count_images_in_dir(RAW_DIR)
    gen_counts = count_images_in_dir(GEN_DIR)

    print("\n--- Rezumat Date ---")
    print(f"Reale: {real_counts}")
    print(f"Generate: {gen_counts}")

    # 3. Configurare Grafic
    x = np.arange(len(CATEGORIES))  # Pozițiile etichetelor
    width = 0.35  # Lățimea barelor

    plt.figure(figsize=(12, 7))
    
    # Desenăm barele
    # Barele reale deplasate la stânga, cele generate la dreapta
    rects1 = plt.bar(x - width/2, real_counts, width, label='Reale (Original)', color='skyblue', edgecolor='black')
    rects2 = plt.bar(x + width/2, gen_counts, width, label='Generate (Sintetice)', color='orange', edgecolor='black')

    # 4. Elemente Grafice
    plt.ylabel('Număr de Imagini', fontsize=12)
    plt.title('Comparație Cantitativă: Imagini Reale vs. Imagini Generate', fontsize=14, fontweight='bold')
    plt.xticks(x, CATEGORIES, fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 5. Adăugarea etichetelor cu numere deasupra barelor
    def autolabel(rects):
        """Atașează o etichetă cu text deasupra fiecărei bare."""
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # 6. Salvare
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"\n✅ Graficul a fost salvat cu succes în: {OUTPUT_FILE}")
    plt.close()

if __name__ == "__main__":
    main()