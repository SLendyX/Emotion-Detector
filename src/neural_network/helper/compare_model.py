import pandas as pd
import matplotlib.pyplot as plt
import os

# Căile către fișiere
custom_path = "results/training_history.csv"
resnet_path = "results/resnet_history.csv"

def compare():
    if not os.path.exists(custom_path) or not os.path.exists(resnet_path):
        print("⚠️ Nu am găsit ambele fișiere CSV. Rulează ambele antrenamente mai întâi!")
        return

    df_custom = pd.read_csv(custom_path)
    df_resnet = pd.read_csv(resnet_path)

    # 1. Plot Comparativ Acuratețe
    plt.figure(figsize=(10, 6))
    plt.plot(df_custom['val_acc'], label='Custom CNN', linestyle='--', color='blue')
    plt.plot(df_resnet['val_acc'], label='ResNet18 (Pretrained)', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Comparison: Custom CNN vs ResNet18')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("docs/grafice/comparison_accuracy.png")
    print("✅ Grafic salvat în docs/grafice/comparison_accuracy.png")

    # 2. Generare Date pentru Tabel
    best_custom = df_custom['val_acc'].max()
    best_resnet = df_resnet['val_acc'].max()
    
    epoch_best_custom = df_custom['val_acc'].idxmax() + 1
    epoch_best_resnet = df_resnet['val_acc'].idxmax() + 1

    print("\n" + "="*40)
    print("      TABEL COMPARATIV (DATE)      ")
    print("="*40)
    print(f"{'Arhitectură':<20} | {'Best Val Acc':<15} | {'Epoch':<10}")
    print("-" * 50)
    print(f"{'Custom CNN':<20} | {best_custom:.2f}%          | {epoch_best_custom}")
    print(f"{'ResNet18':<20} | {best_resnet:.2f}%          | {epoch_best_resnet}")
    print("-" * 50)
    
    diff = best_resnet - best_custom
    if diff > 0:
        print(f"🚀 ResNet este cu {diff:.2f}% mai bun.")
    else:
        print(f"🏆 Modelul tău Custom este cu {abs(diff):.2f}% mai bun (sau similar).")

if __name__ == "__main__":
    compare()