# Emotion Classification CNN

Proiect de clasificare a emoțiilor folosind un model CNN (Convolutional Neural Network) implementat în PyTorch.

## 📋 Cuprins
- [Descriere Generală](#descriere-generală)
- [Arhitectura Modelului](#arhitectura-modelului)
- [Dataset](#dataset)
- [Preprocessing și Augmentare](#preprocessing-și-augmentare)
- [Training Setup](#training-setup)
- [Structura Fișierelor](#structura-fișierelor)
- [Utilizare](#utilizare)

## Descriere Generală

Modelul este proiectat pentru a clasifica imagini faciale în 7 categorii de emoții:
- Angry (supărat)
- Disgust (dezgust)
- Fear (frică)
- Happy (fericit)
- Neutral (neutru)
- Sad (trist)
- Surprised (surprins)

## Arhitectura Modelului

### SimpleEmotionCNN

Modelul este un CNN cu 3 blocuri convoluționale urmate de un layer fully-connected pentru clasificare.

#### Specificații Input/Output
- **Input**: Imagini RGB de dimensiune `100x100x3`
- **Output**: Vector de probabilități pentru 7 clase

#### Structura Detaliată

```
INPUT: [batch_size, 3, 100, 100]
    ↓
LAYER 1:
    - Conv2D: 3 → 32 canale, kernel 3x3, padding 1
    - BatchNorm2D(32)
    - ReLU
    - MaxPool2D: kernel 2x2, stride 2
    Output: [batch_size, 32, 50, 50]
    ↓
LAYER 2:
    - Conv2D: 32 → 64 canale, kernel 3x3, padding 1
    - BatchNorm2D(64)
    - ReLU
    - MaxPool2D: kernel 2x2, stride 2
    Output: [batch_size, 64, 25, 25]
    ↓
LAYER 3:
    - Conv2D: 64 → 128 canale, kernel 3x3, padding 1
    - BatchNorm2D(128)
    - ReLU
    - MaxPool2D: kernel 2x2, stride 2
    Output: [batch_size, 128, 12, 12]
    ↓
FLATTEN: [batch_size, 18432]  (128 * 12 * 12)
    ↓
FULLY CONNECTED: 18432 → 7
    ↓
OUTPUT: [batch_size, 7]
```

#### Parametri Model
- **Total parametri**: ~9.2M
- **Parametri antrenabili**: ~9.2M

### Componente Cheie

1. **Convolutional Blocks**: Extrag feature-uri ierarhice din imagini
2. **Batch Normalization**: Stabilizează training-ul și accelerează convergența
3. **ReLU Activation**: Introduce non-linearitate
4. **Max Pooling**: Reduce dimensionalitatea spațială și oferă invarianță la translatii mici

## Dataset

### Surse de Date

Modelul folosește două surse de date:

1. **Date Reale (RAF-DB)**
   - Locație: `data/raw/train/` și `data/raw/test/`
   - Ponderea în training: 60%

2. **Date Generate (Augmentate/Sintetice)**
   - Locație: `data/generated/`
   - Ponderea în training: 40%

### Structura Directoarelor

```
data/
├── raw/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprised/
│   └── test/
│       └── [aceleași categorii]
└── generated/
    └── [aceleași categorii]
```

### Class Mapping

```python
class_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6
}
```

## Preprocessing și Augmentare

### Training Transforms

Aplicat pe setul de antrenament pentru a crește variabilitatea datelor:

```python
- Resize la 100x100 pixeli
- Random Horizontal Flip (p=0.5)
- Random Rotation (±15 grade)
- Color Jitter:
    - Brightness: ±20%
    - Contrast: ±20%
    - Saturation: ±20%
    - Hue: ±10%
- ToTensor
- Normalize (ImageNet mean & std)
```

### Validation Transforms

Aplicat pe setul de validare (fără augmentare):

```python
- Resize la 100x100 pixeli
- ToTensor
- Normalize (ImageNet mean & std)
```

### Normalizare

Folosește valorile standard de la ImageNet:
- **Mean**: `[0.485, 0.456, 0.406]`
- **Std**: `[0.229, 0.224, 0.225]`

## Training Setup

### Hiperparametri

```python
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
```

### Balansarea Datelor

Utilizează **WeightedRandomSampler** pentru a asigura o distribuție echilibrată:
- 60% șanse de a selecta sample-uri din datele reale
- 40% șanse de a selecta sample-uri din datele generate

Acest lucru previne bias-ul către sursa de date mai mare și asigură că modelul învață din ambele surse.

### Funcție de Loss

```python
CrossEntropyLoss
# Opțional: label_smoothing=0.1 (comentat)
```

### Optimizer

```python
Adam(lr=0.001)
```

### Learning Rate Scheduler

```python
StepLR(step_size=25, gamma=0.1)
# Scade learning rate-ul cu factor de 10 după 25 de epoci
```

Alternative comentate în cod:
- CosineAnnealingLR

### Regularizare

- **Batch Normalization**: În fiecare bloc convolutional
- **Dropout**: Comentat în cod (p=0.5 după flatten)

## Metrici de Evaluare

Pe parcursul antrenamentului se monitorizează:

1. **Training Loss**: Loss-ul mediu pe setul de antrenament
2. **Training Accuracy**: Acuratețea pe setul de antrenament
3. **Validation Loss**: Loss-ul mediu pe setul de validare
4. **Validation Accuracy**: Acuratețea pe setul de validare

### Salvare Checkpoints

- Checkpoint-uri salvate la fiecare 10 epoci
- Locație: `models/latest_checkpoints/emotion_model_epoch_{epoch}.pt`

### Vizualizare

La final se generează grafice pentru:
- Evoluția acurateței (train vs validation)
- Evoluția loss-ului (train vs validation)

Graficele se salvează în: `docs/grafice/training_curves.png`

## Structura Fișierelor

```
.
├── my_training.py          # Script principal de training
├── data/
│   ├── raw/
│   │   ├── train/         # Date de antrenament originale
│   │   └── test/          # Date de testare originale
│   └── generated/         # Date generate/augmentate
├── models/
│   └── latest_checkpoints/ # Checkpoints salvate
└── docs/
    └── grafice/           # Grafice de training
```

## Utilizare

### Cerințe

```bash
pip install torch torchvision pillow pandas numpy matplotlib
```

### Training

```bash
python my_training.py
```

### Device

Scriptul detectează automat dacă CUDA este disponibil:
- Folosește GPU dacă este disponibil
- Fallback pe CPU altfel

## Îmbunătățiri Potențiale

Comentate în cod, dar pot fi activate:

1. **Label Smoothing**: `CrossEntropyLoss(label_smoothing=0.1)`
2. **Dropout**: `nn.Dropout(p=0.5)` după flatten
3. **CosineAnnealingLR**: Scheduler alternativ pentru learning rate
4. **Transfer Learning**: Folosirea unui model pre-antrenat (ResNet, EfficientNet)
5. **Class Weights**: Pentru datasets cu clase dezechilibrate
6. **Early Stopping**: Pentru a preveni overfitting-ul

## Observații

- Modelul folosește arhitectură simplă, potrivită pentru proof-of-concept
- Batch Normalization ajută la stabilizarea training-ului
- WeightedRandomSampler asigură că datele generate nu domină training-ul
- Augmentarea pe training set ajută la generalizare

---

**Data ultimei actualizări**: Februarie 2026  
**Framework**: PyTorch  
**Tip Model**: Convolutional Neural Network (CNN)