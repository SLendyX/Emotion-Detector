# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Ion Radu-Stefan  
**Link Repository GitHub:** [https://github.com/SLendyX/Emotion-Detector](https://github.com/SLendyX/Emotion-Detector)  
**Data:** 04.12.2025  

---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

### IMPORTANT - Ce înseamnă "schelet funcțional":

**CE TREBUIE SĂ FUNCȚIONEZE:**
- Toate modulele pornesc fără erori.
- Pipeline-ul complet rulează end-to-end (de la date → până la output UI).
- Modelul RN este definit și compilat (arhitectura există).
- Web Service/UI primește input și returnează output.

 **CE NU E NECESAR ÎN ETAPA 4:**
- Model RN antrenat cu performanță bună
- Hiperparametri optimizați
- Acuratețe mare pe test set
- Web Service/UI cu funcționalități avansate

**Scopul anti-plagiat:** Nu puteți copia un notebook + model pre-antrenat de pe internet, pentru că modelul vostru este NEANTRENAT în această etapă. Demonstrați că înțelegeți arhitectura și că ați construit sistemul de la zero.

---

## 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă**                                  | **Cum o rezolvă SIA-ul vostru**                                                     | **Modul software responsabil** |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------ |
| Monitorizarea stării de bine a angajaților în regim remote | Detecție emoții + Estimare puls (rPPG) → Identificare semne de stres în < 1 secundă | RN + Heart Rate Module + UI    |
| Evaluarea reacției utilizatorilor la conținut digital (UX) | Clasificare automată a micro-expresiilor faciale → Raport impact emoțional          | RN + Data Logging              |
| Securitate asistată prin analiză comportamentală           | Detectarea agitației prin corelarea pulsului cu mimica facială                      | RN + Web Service               |

---

## 2. Contribuția originală la setul de date

**Total observații finale:** [N] (după Etapa 3 + Etapa 4)  
**Observații originale:** [M] (minim 40% conform cerinței)  

**Tipul contribuției:** 
[] Date generate prin simulare fizică  
[X] Date achiziționate cu senzori proprii (Webcam)  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:** Contribuția originală constă în utilizarea scriptului `capture_data.py` pentru a achiziția de imagini faciale direct de la webcam în condiții de iluminare variate. Această metodă a fost aleasă pentru a asigura robustețea modelului în scenariul de utilizare "live", adaptând sistemul la trăsăturile specifice ale utilizatorului și la calitatea senzorului local.

Procesul a implicat detectarea feței, decuparea automată (crop) și redimensionarea la 48x48 pixeli în format grayscale pentru a menține consistența cu restul dataset-ului.

**Locația codului:** `src/data_acquisition/capture_data.py`  
**Locația datelor:** `data/generated/`  

**Dovezi:** - Grafic comparativ: `docs/generated_vs_real.png`  
- Setup experimental: `docs/acquisition_setup.jpg`  

---

## 3. Diagrama State Machine (Mașina de Stări)

![State Machine Diagram](../Schema_functionare.drawio%20(1).png)  
*(Fișierul se află în docs/)*

### Justificarea State Machine-ului ales:

Am ales o arhitectură de **monitorizare continuă** deoarece proiectul necesită procesarea unui flux video în timp real pentru a extrage simultan date despre emoții și puls.

**Stările principale sunt:**
1. **IDLE / Initialization:** Sistemul încarcă modelul Rețelei Neuronale (`emotion_model.pt`) și inițializează buffer-ul pentru puls.
2. **ACQUISITION:** Preluarea continuă a cadrelor video de la webcam.
3. **DETECTION & PROCESSING:** Identificarea feței; dacă se detectează o față, se trece la pre-procesarea imaginii pentru RN și analiza culorii pentru rPPG.
4. **INFERENCE:** Execuția modelului CNN pentru emoții și calculul FFT pentru puls.
5. **DISPLAY / FEEDBACK:** Rezultatele sunt agregate și afișate utilizatorului sub formă de diagnostic (ex: Stres, Relaxare).

Tranziția către starea de **ERROR** este esențială pentru a gestiona situațiile în care camera este deconectată sau modelul nu poate fi încărcat.

---

## 4. Scheletul Complet al celor 3 Module

| **Modul**                         | **Tehnologie**                     | **Cerință minimă funcțională**                                            |
| --------------------------------- | ---------------------------------- | ------------------------------------------------------------------------- |
| **1. Data Logging / Acquisition** | Python (`collect_highdef_data.py`) | Produce imagini cu datele originale (40%) și rulează fără erori.          |
| **2. Neural Network Module**      | Pytorch/Torchvision (`train.py`)   | Modelul CNN este definit, compilat și poate fi încărcat pentru inferență. |
| **3. Web Service / UI**           | OpenCV / `color_app_web.py`        | Primește input video și afișează predicția emoției și a pulsului.         |

---

## 5. Structura Repository-ului (Etapa 4)