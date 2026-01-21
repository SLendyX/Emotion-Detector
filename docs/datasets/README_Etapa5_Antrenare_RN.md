# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ion Radu-Stefan  
**Link Repository GitHub:** [Adaugă link-ul tău aici]  
**Data predării:** 20.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN definit în Etapa 4, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Logging, RN, UI)
- Minimum 40% date originale în dataset

---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/state_machine.png`
- [x] **Contribuție ≥40% date originale** în `data/generated/` (integrat prin `preprocess_data.py`)
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [x] **Modul 2 (RN)** cu arhitectură definită (`src/neural_network/train.py`)
- [x] **Modul 3 (UI/Web Service)** funcțional (`src/ui/main_app.py`)
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

---

## Pregătire Date pentru Antrenare 

### Dacă ați adăugat date noi în Etapa 4 (contribuția de 40%):

Dataset-ul a fost recombinat și balansat dinamic folosind scriptul `src/preprocessing/preprocess_data.py`.
S-a realizat mixarea datelor generate proprii (40%) cu datele din setul public FER2013 (60%).

**Structura finală a datelor:**
- **Train/Validation:** Conține date mixte (Proprii + FER2013) pentru a asigura învățarea trăsăturilor specifice utilizatorului.
- **Test:** Conține **exclusiv** date publice (FER2013) pentru o evaluare obiectivă a generalizării.

**Parametri utilizați:**
- Scalare: Normalizare pixel [0, 1] (împărțire la 255.0).
- Split: 85% Train / 15% Validation (stratificat).
- Random State: 42.

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

1. **Antrenare model:** Realizată pe setul final folosind `src/neural_network/train.py`.
2. **Epoci:** Setat la 60 epoci, cu oprire automată (Early Stopping) la epoca 58.
3. **Împărțire:** Stratificată (Train/Val/Test).
4. **Metrici calculate pe test set:** (Conform `results/test_metrics.json`)
   - **Acuratețe Test:** 43.70% (Notă: Acuratețea pe validare a fost ~68%, diferența provine din faptul că setul de test este pur FER2013, mult mai dificil).
   - **F1-score (macro):** 0.4291
5. **Salvare model:** `models/trained_model.h5`.
6. **Integrare UI:** Modelul este încărcat și utilizat în `src/ui/main_app.py`.

#### Tabel Hiperparametri și Justificări (OBLIGATORIU - Nivel 1)

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.0003 | Valoare redusă pentru stabilitate, ajustată dinamic de scheduler (`ReduceLROnPlateau`). |
| Batch size | 64 | Crescut pentru o estimare mai stabilă a gradientului și viteză mai mare pe datele augmentate. |
| Number of epochs | 70 | Extins pentru a permite convergență lentă cu `EarlyStopping` (patience=12). |
| Optimizer | Adam | Algoritm adaptiv eficient pentru arhitecturi CNN. |
| Loss function | Categorical Crossentropy (Label Smoothing 0.1) | Label Smoothing ajută modelul să fie mai puțin "supra-încrezător" pe clasele similare (ex: Fear vs Surprise). |
| Class Weights | Manual (Angry:1.25, Digust:2.0...) | Balansare costuri pentru a penaliza mai mult erorile pe clasele minoritare (Disgust). |
| Regularization (L2) | 0.0001 | L2 Penalty pe stratul Dense final pentru a preveni overfitting-ul pe trăsături zgomotoase. |
| Dropout | 0.2, 0.3, 0.4, 0.5 | Dropout progresiv (0.2 -> 0.5) pentru a forța o generalizare robustă în straturile adânci. |

---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1. **Early Stopping:** Implementat în `train.py` cu `patience=10`. Monitorizează `val_loss`.
2. **Learning Rate Scheduler:** Folosit `ReduceLROnPlateau` (factor=0.5, patience=4).
3. **Augmentări relevante:** - Implementate în `train.py` folosind `ImageDataGenerator`.
   - Rotații (20°), Shift (10%), Zoom (20%), Shear (10%), Horizontal Flip.
4. **Grafic loss și val_loss:** Salvat în `docs/loss_curve.png`.
5. **Analiză erori context industrial:** Vezi secțiunea de mai jos.

---

### Nivel 3 – Bonus (până la 100%)

**Activități realizate:**

| **Activitate** |  **Livrabil** |
|----------------|--------------|
| Confusion Matrix + analiză | `docs/confusion_matrix.png` generat de `src/analysis/evaluate.py`. Analiza detaliată este prezentată mai jos. |

---

## Verificare Consistență cu State Machine (Etapa 4)

Fluxul din `main_app.py` respectă diagrama:

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | `cap.read()` preia frame-ul în buclă infinită. |
| `PREPROCESS` | Funcția `preprocess_face()` convertește la Grayscale și normalizează (div 255). |
| `RN_INFERENCE` | `model.predict(processed)` rulează pe modelul încărcat din `trained_model.h5`. |
| `THRESHOLD_CHECK` | Se verifică `bpm > 100` sau Emoție = Fear/Angry pentru a determina starea de alertă. |
| `ALERT` | UI-ul schimbă culoarea chenarului în ROȘU și afișează "Zona: STRES / ALERTA". |

Cod în `src/ui/main_app.py`:
```python
# Modelul antrenat este încărcat
model = tf.keras.models.load_model('models/trained_model.h5')
# ...
pred = model.predict(processed, verbose=0) # Inferență reală
emotion_idx = np.argmax(pred)