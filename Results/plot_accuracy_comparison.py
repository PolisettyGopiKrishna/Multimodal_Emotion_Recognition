import pandas as pd
import matplotlib.pyplot as plt

# Load accuracy files
speech = pd.read_csv("speech_accuracy.csv")
text = pd.read_csv("text_accuracy.csv")
fusion = pd.read_csv("fusion_accuracy.csv")

df = pd.concat([speech, text, fusion])

plt.figure(figsize=(7,5))
bars = plt.bar(df["Model"], df["Accuracy"])

plt.title("Accuracy Comparison: Speech vs Text vs Fusion")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

# Add accuracy values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height,
             f"{height:.2f}",
             ha='center',
             va='bottom')

plt.savefig("accuracy_comparison.png")
plt.show()

print("Accuracy comparison plot saved successfully.")
