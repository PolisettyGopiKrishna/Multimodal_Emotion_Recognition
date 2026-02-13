# Multimodal Emotion Recognition

## 📌 Objective

This project builds a Multimodal Emotion Recognition system using:

- 🎤 Speech only
- 📝 Text only
- 🔗 Fusion of Speech + Text

The goal is to compare the performance of unimodal and multimodal systems using the Toronto Emotional Speech Set (TESS).

---

## 📂 Dataset

**Dataset Used:** Toronto Emotional Speech Set (TESS)

The dataset contains:
- Speech audio files (.wav)
- Corresponding transcript words (derived from filename)
- Emotion labels

Emotions include:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise
(Depending on dataset version)

Place dataset inside:


---

## 🏗️ Project Structure

project/
│
├── data/
│
├── models/
│ ├── speech_pipeline/
│ ├── text_pipeline/
│ └── fusion_pipeline/
│
├── Results/
│ ├── speech_accuracy.csv
│ ├── text_accuracy.csv
│ ├── fusion_accuracy.csv
│ ├── accuracy_comparison.png
│ └── plots/
│
├── README.md
└── requirements.txt

---

## ⚙️ Architecture Design

### 1️⃣ Speech Pipeline

- Preprocessing:
  - Resampling to 16kHz
  - Silence trimming
- Feature Extraction:
  - MFCC (40 coefficients)
- Temporal Modelling:
  - BiLSTM (Bidirectional LSTM)
- Classifier:
  - Fully Connected layer

---

### 2️⃣ Text Pipeline

- Text extraction from filename
- Tokenization using BERT tokenizer
- Contextual Modelling:
  - Pretrained BERT (bert-base-uncased)
- Classifier:
  - Linear layer on CLS token

---

### 3️⃣ Fusion Pipeline

- Speech embedding (256-dim)
- Text embedding (768-dim)
- Concatenation (1024-dim)
- Fully Connected classifier

Fusion learns joint representation from both modalities.

---

## 🚀 How to Run

### 1️⃣ Install dependencies

pip install -r requirements.txt

---

### 2️⃣ Train Models

Speech:
cd models/speech_pipeline
python train.py
Text:
cd models/text_pipeline
python train.py

Fusion:
cd models/fusion_pipeline
python train.py

---

### 3️⃣ Test Models

Speech:
python test.py

Text:
python test.py

Fusion:
python test.py

Accuracy Comparison:
cd Results
python plot_accuracy_comparison.py

---

## 📊 Results

The system evaluates:

- Test Accuracy
- Confusion Matrix
- Accuracy comparison bar plot

### Observations

- Text model performs strongly due to contextual understanding from BERT.
- Speech model captures emotional tone patterns.
- Fusion improves performance by combining acoustic and semantic information.
- Fusion particularly improves classification for subtle emotions.

---

## 📈 Visualization

Generated:

- Speech confusion matrix
- Text confusion matrix
- Fusion confusion matrix
- Accuracy comparison plot

Fusion embeddings show better class separability.

---

## 🧠 Key Insights

- Speech captures prosody and tone.
- Text captures semantic meaning.
- Fusion helps when one modality is ambiguous.
- Hardest emotions: Fear vs Surprise (similar acoustic features)
- Easiest emotions: Angry, Happy (distinct patterns)

---

## 🛠️ Libraries Used

- PyTorch
- Transformers (HuggingFace)
- Librosa
- Scikit-learn
- Matplotlib
- Seaborn

---

