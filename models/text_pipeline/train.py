import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DATA_PATH = "../../data/TESS Toronto emotional speech set data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Load Text + Labels
# ------------------------
texts, emotions = [], []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            text = file.split("_")[0]
            texts.append(text)
            emotions.append(root.split("_")[-1])

le = LabelEncoder()
labels = le.fit_transform(emotions)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ------------------------
# Dataset Class
# ------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=16,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_loader = DataLoader(
    TextDataset(X_train, y_train),
    batch_size=16,
    shuffle=True
)

# ------------------------
# Model
# ------------------------
class TextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls = output.pooler_output
        return self.fc(cls)

model = TextModel(len(le.classes_)).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# ------------------------
# Training Loop
# ------------------------
for epoch in range(3):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "text_model.pth")
print("Model saved successfully.")
