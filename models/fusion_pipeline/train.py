import os
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "../../data/TESS Toronto emotional speech set data"

# -----------------------
# Load Data
# -----------------------
files, texts, emotions = [], [], []

for root, dirs, filenames in os.walk(DATA_PATH):
    for file in filenames:
        if file.endswith(".wav"):
            files.append(os.path.join(root, file))
            texts.append(file.split("_")[0])
            emotions.append(root.split("_")[-1])

le = LabelEncoder()
labels = le.fit_transform(emotions)
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    files, labels, texts, test_size=0.2, random_state=42
)

# -----------------------
# Dataset
# -----------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class FusionDataset(Dataset):
    def __init__(self, files, texts, labels):
        self.files = files
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.files[idx], sr=16000)
        y, _ = librosa.effects.trim(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return mfcc, encoding, torch.tensor(self.labels[idx])

def collate_fn(batch):
    speech, text, labels = zip(*batch)
    speech = nn.utils.rnn.pad_sequence(speech, batch_first=True)
    labels = torch.stack(labels)

    input_ids = torch.stack([t["input_ids"] for t in text])
    attention_mask = torch.stack([t["attention_mask"] for t in text])
    token_type_ids = torch.stack([t["token_type_ids"] for t in text])

    return speech, input_ids, attention_mask, token_type_ids, labels

train_loader = DataLoader(
    FusionDataset(X_train, text_train, y_train),
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# -----------------------
# Encoders (Frozen)
# -----------------------
class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(40, 128, batch_first=True, bidirectional=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.cat((h[-2], h[-1]), dim=1)

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return output.pooler_output

speech_encoder = SpeechEncoder().to(DEVICE)
speech_encoder.load_state_dict(
    torch.load("../speech_pipeline/speech_model.pth", map_location=DEVICE),
    strict=False
)
speech_encoder.eval()

text_encoder = TextEncoder().to(DEVICE)
text_encoder.eval()

for param in speech_encoder.parameters():
    param.requires_grad = False
for param in text_encoder.parameters():
    param.requires_grad = False

# -----------------------
# Fusion Classifier
# -----------------------
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, speech_feat, text_feat):
        combined = torch.cat([speech_feat, text_feat], dim=1)
        return self.fc(combined)

fusion_model = FusionModel(num_classes).to(DEVICE)

optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# -----------------------
# Training
# -----------------------
for epoch in range(5):
    fusion_model.train()
    total_loss = 0

    for speech, input_ids, attention_mask, token_type_ids, labels in tqdm(train_loader):

        speech = speech.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            speech_feat = speech_encoder(speech)
            text_feat = text_encoder(input_ids, attention_mask, token_type_ids)

        outputs = fusion_model(speech_feat, text_feat)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(fusion_model.state_dict(), "fusion_model.pth")
print("Fusion model trained and saved successfully.")
