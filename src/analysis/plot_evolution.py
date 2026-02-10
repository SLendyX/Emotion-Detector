import matplotlib.pyplot as plt
import numpy as np
import os

# --- DATELE PENTRU CELE 3 ETAPE ---
stages = ['Etapa 4\n(MVP/Random)', 'Etapa 5\n(Baseline)', 'Etapa 6\n(Final Optimizat)']

# 1. Acuratețe și F1 (Scara 0 - 100%)
accuracy = [25, 65, 70]   # Valori in procente
f1_score = [20, 60, 70]   # Valori in procente

# 2. Latență (ms) - Scara Inversă (mai puțin e mai bine)
latency = [80, 12, 1.41] 

# --- CONFIGURARE GRAFIC ---
OUTPUT_DIR = "docs/grafice"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    # Setăm stilul
    plt.style.use('ggplot')
    
    # Creăm o figură cu 2 subplot-uri (Performanță vs Viteză)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- SUBPLOT 1: PERFORMANȚĂ (Accuracy & F1) ---
    x = np.arange(len(stages))
    width = 0.35  # Lățimea barelor

    rects1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#3498db')
    rects2 = ax1.bar(x + width/2, f1_score, width, label='F1-Score (%)', color='#2ecc71')

    ax1.set_ylabel('Scor (%)')
    ax1.set_title('Evoluția Performanței (Mai mare e mai bine)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylim(0, 100) # Limita Y la 100%
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Adăugăm etichete cu valori pe bare
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # --- SUBPLOT 2: VITEZĂ (Latență) ---
    # Folosim bar chart simplu, dar colorăm diferit etapa finală pentru impact
    colors = ['#95a5a6', '#f39c12', '#e74c3c'] # Gri, Portocaliu, Roșu intens
    rects3 = ax2.bar(stages, latency, color=colors, width=0.5)

    ax2.set_ylabel('Timp (milisecunde)')
    ax2.set_title('Reducerea Latenței (Mai mic e mai bine)')
    ax2.grid(True, axis='y', alpha=0.3)

    # Adăugăm etichete pe barele de latență
    for rect in rects3:
        height = rect.get_height()
        ax2.annotate(f'{height} ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    # Adăugăm o săgeată curbată pentru a arăta îmbunătățirea masivă
    ax2.annotate('ONNX Optimization\n(-88%)', 
                 xy=(2, 8), xytext=(1.25, 50),
                 arrowprops=dict(facecolor='black', shrink=0.05, connectionstyle="arc3,rad=.2"))

    plt.tight_layout()
    
    # Salvare
    save_path = os.path.join(OUTPUT_DIR, "metrics_evolution.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Graficul evolutiv a fost salvat: {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()