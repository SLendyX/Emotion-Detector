import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- CONFIGURARE ---
DATA_DIR = "data/processed"
DOCS_DIR = "docs"  # Unde salvăm graficele rezultate

# Listă categorii (trebuie să fie în aceeași ordine ca la preprocesare)
CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def main():
    # 1. Încărcăm datele de antrenament
    print("📥 Încărcare date din .npy...")
    try:
        # X_train conține imaginile (matrice de pixeli)
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        # y_train conține etichetele (în format One-Hot: [0, 0, 1, 0...])
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    except FileNotFoundError:
        print("❌ Eroare: Nu găsesc fișierele .npy. Ai rulat preprocesarea?")
        return

    # Convertim etichetele din One-Hot în numere simple (ex: din [0,0,1,0] în 2)
    # np.argmax găsește poziția unde este '1'
    y_indices = np.argmax(y_train, axis=1)

    # Ne asigurăm că folderul de documentație există
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # =========================================================
    # PARTEA A: Distribuția Claselor (Histograma)
    # =========================================================
    print("📊 Generare histogramă...")
    
    # Numărăm câte imagini sunt în fiecare categorie
    counts = []
    for i in range(len(CATEGORIES)):
        # Numărăm de câte ori apare indexul 'i' în y_indices
        count = np.sum(y_indices == i)
        counts.append(count)

    # Creăm graficul
    plt.figure(figsize=(10, 6))
    bars = plt.bar(CATEGORIES, counts, color='skyblue', edgecolor='black')
    
    # Adăugăm numărul exact deasupra fiecărei bare
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.title('Distribuția Claselor de Emoții în Setul de Antrenament')
    plt.xlabel('Emoție')
    plt.ylabel('Număr de Imagini')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Salvăm graficul
    hist_path = os.path.join(DOCS_DIR, "distributie_clase.png")
    plt.savefig(hist_path)
    print(f"✅ Histograma salvată la: {hist_path}")
    plt.close() # Închidem figura pentru a elibera memoria

    # =========================================================
    # PARTEA B: Vizualizare Eșantioane Aleatorii
    # =========================================================
    print("🖼️ Generare vizualizare exemple...")

    plt.figure(figsize=(15, 3)) # Figură lungă

    for i, emotion_name in enumerate(CATEGORIES):
        # Găsim toți indicii imaginilor care au emoția 'i'
        indices_of_emotion = np.where(y_indices == i)[0]
        
        # Alegem un index aleatoriu dintre aceștia
        random_idx = random.choice(indices_of_emotion)
        
        # Extragem imaginea
        img = X_train[random_idx]
        
        # Facem subplot (1 rând, 7 coloane)
        plt.subplot(1, 7, i + 1)
        
        # img are forma (48, 48, 1). Trebuie să scăpăm de ultimul 1 pentru plotare
        # .squeeze() transformă (48, 48, 1) în (48, 48)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(emotion_name)
        plt.axis('off') # Ascundem axele cu numere

    plt.tight_layout()
    samples_path = os.path.join(DOCS_DIR, "esantioane_emotii.png")
    plt.savefig(samples_path)
    print(f"✅ Exemplele salvate la: {samples_path}")
    plt.close()

if __name__ == "__main__":
    main()