
import joblib
from pathlib import Path

MODEL_PATH = Path("model.pkl")

if not MODEL_PATH.exists():
    print("Model not found. First run: python train.py")
    raise SystemExit(1)

model = joblib.load(MODEL_PATH)

print("=" * 55)
print(" STUDENT FEEDBACK & TASK PRIORITY CATEGORIZER")
print("=" * 55)
print("Enter student feedback to classify it.")
print("Type 'exit' to close the program.\n")

while True:
    feedback = input("Enter feedback: ").strip()

    if feedback.lower() == "exit":
        print("Program closed. Goodbye!")
        break

    if not feedback:
        print("Please enter some feedback.\n")
        continue

    prediction = model.predict([feedback])[0]

    print(f"\nPredicted Category: {prediction}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([feedback])[0]
        confidence = max(probabilities) * 100
        print(f"Model confidence: {confidence:.1f}%")

    print("-" * 55)
