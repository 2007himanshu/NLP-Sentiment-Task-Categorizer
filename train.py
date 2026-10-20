
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load the dataset
data = pd.read_csv("data.csv")

# Check required columns
if not {"text", "label"}.issubset(data.columns):
    raise ValueError("data.csv must contain 'text' and 'label' columns.")

data = data.dropna(subset=["text", "label"])
data["text"] = data["text"].astype(str)
data["label"] = data["label"].astype(str)

print("Dataset loaded successfully!")
print("\nExamples per category:")
print(data["label"].value_counts())

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.25,
    random_state=42,
    stratify=data["label"]
)

# 3. Create the NLP pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000, random_state=42)
    )
])

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate the model
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# 6. Create and save the confusion matrix
labels = sorted(data["label"].unique())
matrix = confusion_matrix(y_test, predictions, labels=labels)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)
display.plot(xticks_rotation=25)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 7. Save the trained model
joblib.dump(model, "model.pkl")

print("\nModel saved as model.pkl")
print("Confusion matrix saved as confusion_matrix.png")
print("\nTraining completed!")
