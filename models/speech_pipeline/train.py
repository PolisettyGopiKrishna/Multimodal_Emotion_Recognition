import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DATA_PATH = "../../data/TESS Toronto emotional speech set data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SpeechDataset(Dataset):
    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        y, sr = librosa.load(file, sr=16000)
        y, _ = librosa.effects.trim(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

        return mfcc, torch.tensor(self.labels[idx])

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    return xs.to(DEVICE), torch.stack(ys).to(DEVICE)

class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(40, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

# Load data
files, emotions = [], []

for root, dirs, filenames in os.walk(DATA_PATH):
    for file in filenames:
        if file.endswith(".wav"):
            files.append(os.path.join(root, file))
            emotions.append(root.split("_")[-1])

le = LabelEncoder()
labels = le.fit_transform(emotions)

X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.2)

train_loader = DataLoader(SpeechDataset(X_train, y_train),
                          batch_size=16,
                          shuffle=True,
                          collate_fn=collate_fn)

model = SpeechModel(len(le.classes_)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(15):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "speech_model.pth")
