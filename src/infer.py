from pathlib import Path

import joblib


def predict_sentiment(text: str, model_path: str = "models/sentiment_model.joblib") -> tuple[str, float]:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found at {model_file}. Run `python src/train.py` first."
        )

    model = joblib.load(model_file)
    probs = model.predict_proba([text])[0]
    classes = model.classes_

    best_idx = probs.argmax()
    label = str(classes[best_idx])
    confidence = float(probs[best_idx])
    return label, confidence


if __name__ == "__main__":
    while True:
        msg = input("Enter text (or 'quit'): ").strip()
        if msg.lower() == "quit":
            break
        sentiment, conf = predict_sentiment(msg)
        print(f"Sentiment: {sentiment} (confidence={conf:.3f})")
