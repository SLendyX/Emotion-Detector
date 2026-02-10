import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Calea catre rezultate
path = "results/experiments"
all_files = glob.glob(os.path.join(path, "*.csv"))

plt.figure(figsize=(12, 6))

for filename in all_files:
    df = pd.read_csv(filename)
    name = os.path.basename(filename).replace("_history.csv", "")
    
    # Plot doar Validation Accuracy (cea care conteaza)
    plt.plot(df['epoch'], df['val_acc'], label=name, linewidth=2)
    
    # Afiseaza max acc in legenda
    max_acc = df['val_acc'].max()
    print(f"{name}: Best Accuracy = {max_acc:.2f}%")

plt.title("Comparatie Experimente (Validation Accuracy)")
plt.xlabel("Epoci")
plt.ylabel("Acuratețe (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("docs/grafice/comparison_experiments.png")
print("✅ Grafic salvat in docs/grafice/comparison_experiments.png")