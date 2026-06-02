import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import json

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

data["content"] = data["title"] + " " + data["text"]

x = data["content"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english")
x = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

metrics = {
    "accuracy": float(accuracy),
    "training_samples": int(len(data)),
    "real_samples": int(true.shape[0]),
    "fake_samples": int(fake.shape[0]),
    "tn": int(cm[0, 0]),
    "fp": int(cm[0, 1]),
    "fn": int(cm[1, 0]),
    "tp": int(cm[1, 1])
}

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Model trained successfully!")