import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "../../data/TESS Toronto emotional speech set data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

test_loader = DataLoader(
    TextDataset(X_test, y_test),
    batch_size=16,
    shuffle=False
)

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
        return self.fc(output.pooler_output)

model = TextModel(len(le.classes_)).to(DEVICE)
model.load_state_dict(torch.load("text_model.pth", map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask, token_type_ids)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\nText Test Accuracy: {accuracy:.4f}\n")

pd.DataFrame({
    "Model": ["Text"],
    "Accuracy": [accuracy]
}).to_csv("../../Results/text_accuracy.csv", index=False)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.title("Text Confusion Matrix")
plt.savefig("../../Results/plots/text_confusion_matrix.png")
plt.close()

print("Text results saved.")
