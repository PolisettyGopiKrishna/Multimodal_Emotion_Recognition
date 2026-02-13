import os
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "../../data/TESS Toronto emotional speech set data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load Data
# ------------------------
files, emotions = [], []

for root, dirs, filenames in os.walk(DATA_PATH):
    for file in filenames:
        if file.endswith(".wav"):
            files.append(os.path.join(root, file))
            emotions.append(root.split("_")[-1])

le = LabelEncoder()
labels = le.fit_transform(emotions)

X_train, X_test, y_train, y_test = train_test_split(
    files, labels, test_size=0.2, random_state=42
)

# ------------------------
# Dataset
# ------------------------
class SpeechDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.files[idx], sr=16000)
        y, _ = librosa.effects.trim(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

        return mfcc, torch.tensor(self.labels[idx])

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    return xs.to(DEVICE), torch.stack(ys).to(DEVICE)

test_loader = DataLoader(
    SpeechDataset(X_test, y_test),
    batch_size=16,
    shuffle=False,
    collate_fn=collate_fn
)

# ------------------------
# Model
# ------------------------
class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(40, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

model = SpeechModel(len(le.classes_)).to(DEVICE)
model.load_state_dict(torch.load("speech_model.pth", map_location=DEVICE))
model.eval()

# ------------------------
# Testing
# ------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nSpeech Test Accuracy: {accuracy:.4f}\n")

print(classification_report(all_labels, all_preds, target_names=le.classes_))

# Save results
os.makedirs("../../Results/plots", exist_ok=True)

pd.DataFrame({
    "Model": ["Speech"],
    "Accuracy": [accuracy]
}).to_csv("../../Results/speech_accuracy.csv", index=False)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.title("Speech Confusion Matrix")
plt.savefig("../../Results/plots/speech_confusion_matrix.png")
plt.close()

print("Speech results saved.")
