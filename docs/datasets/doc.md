##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** [Dataset-ul FER2013 (Kaggle Challenge)](https://www.kaggle.com/datasets/msambare/fer2013)
* **Modul de achiziție:** Fișier extern (descărcat)
* **Perioada / condițiile colectării:** Date istorice, colectate pentru competiția ICML 2013

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** Aprox. 35,887 imagini
* **Număr de caracteristici (features):** 2304 (48x48 pixeli) + 1 etichetă (emoția)
* **Tipuri de date:** Imagini (convertite în valori numerice de pixeli) și Categoriale (eticheta emoției)
* **Format fișiere:** CSV

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| pixels | numeric (matrice) | intensitate | Valorile pixelilor imaginii grayscale (48x48) | 0-255 |
| emotion | categorial | - | Clasa emoției (0=Furie, 1=Dezgust, 2=Frică, 3=Fericire, 4=Tristețe, 5=Surpriză, 6=Neutru) | 0–6 |
| usage | categorial | - | Indică dacă exemplul este pentru Training/PublicTest/PrivateTest | {Training, Test} |


**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie, mediană, deviație standard** 
![histograma pixeli](../histograma_pixeli.png)
*Fig 1. Distribuția globală a intensității pixelilor. Forma de clopot indică o normalizare corectă.*

  * Medie: 0.5077 (Ideal ~0.5 pentru date normalizate)
  * Mediană: 0.5255
  * Deviație Standard: 0.2551

* **Min–max și quartile**
  * Min: 0.0, Max: 1.0
  * Q1 (25%): 0.3098
  * Q3 (75%): 0.7098

* **Distribuții pe caracteristici** (histograme)
![distributie clase emotii](../distributie_clase.png)
*Fig 2. Dezechilibrul claselor în setul de antrenare. Se observă predominanța clasei "Happy".*

* **Identificarea outlierilor** (IQR / percentile)
![boxplot outlieri](../boxplot_outlieri.png)
*Fig 3. Identificarea imaginilor extreme (prea întunecate sau prea luminoase) folosind metoda IQR.*

  * IQR (Interquartile Range): 0.1799
  * Limita inferioară (Prea întunecat): 0.1491
  * Limita superioară (Prea luminos): 0.8687
  * Imagini outlier întunecate: 75
  * Imagini outlier luminoase: 45

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** (% pe coloană)
  * 0% valori lipsă: Nu s-au identificat valori nule (NaN) sau pixeli lipsă în matricele de imagini. Procesul de prelucrare a asigurat convertirea tuturor imaginilor valide în format numeric. Imaginile corupte (care nu au putut fi citite de OpenCV) au fost excluse automat în etapa de preprocesare.

* **Detectarea valorilor inconsistente sau eronate**
  * Outlieri de luminozitate: Folosind metoda IQR (Interquartile Range) pe luminozitatea medie a imaginilor, s-au identificat 120 de imagini potențial problematice:

    * 75 imagini "întunecate" (medie < 0.1491): Acestea sunt aproape negre și pot fi erori de captură sau ocluzii severe.

    * 45 imagini "supraexpuse" (medie > 0.8687): Imagini aproape albe, unde trăsăturile faciale pot fi pierdute.

  * Dezechilibru major de clase (Class Imbalance): Analiza distribuției claselor indică o inconsistență majoră în reprezentarea emoțiilor.

    * Clasa Happy este supra-reprezentată (7215 imagini).

    * Clasa Disgust este extrem de sub-reprezentată (doar 436 imagini).

    * Impact: Modelul riscă să învețe foarte bine să detecteze fericirea, dar să eșueze în detectarea dezgustului.

* **Identificarea caracteristicilor redundante sau puternic corelate**
  * Corelație spațială: În cazul datelor de tip imagine, pixelii adiacenți prezintă o corelație puternică (valori similare în vecinătate). Aceasta nu este considerată o redundanță negativă, ci o proprietate esențială pe care arhitectura CNN (Rețea Neuronală Convoluțională) o va exploata pentru a detecta contururi și forme.

  * Nu există coloane redundante (toți cei 2304 pixeli contribuie la imaginea de ansamblu).

### 3.3 Probleme identificate

* [exemplu] Feature X are 8% valori lipsă
* [exemplu] Distribuția feature Y este puternic neuniformă
* [exemplu] Variabilitate ridicată în clase (class imbalance)

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**
* **Tratarea valorilor lipsă:**
  * Feature A: imputare cu mediană
  * Feature B: eliminare (30% valori lipsă)
* **Tratarea outlierilor:** IQR / limitare percentile

### 4.2 Transformarea caracteristicilor

* **Normalizare:** Min–Max / Standardizare
* **Encoding pentru variabile categoriale**
* **Ajustarea dezechilibrului de clasă** (dacă este cazul)

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [X] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---
