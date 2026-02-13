from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def load_split(path: Path) -> tuple[pd.Series, pd.Series]:
    df = pd.read_csv(path)
    return df["text"], df["label"]


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )


def main() -> None:
    data_dir = Path("data")
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(data_dir / "train.csv")
    x_val, y_val = load_split(data_dir / "val.csv")
    x_test, y_test = load_split(data_dir / "test.csv")

    model = build_model()
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    print("Validation metrics:")
    print(classification_report(y_val, val_pred, digits=4))
    print("Validation confusion matrix:")
    print(confusion_matrix(y_val, val_pred, labels=["negative", "neutral", "positive"]))

    print("\nTest metrics:")
    print(classification_report(y_test, test_pred, digits=4))
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_pred, labels=["negative", "neutral", "positive"]))

    joblib.dump(model, model_dir / "sentiment_model.joblib")
    print(f"\nSaved model to: {model_dir / 'sentiment_model.joblib'}")


if __name__ == "__main__":
    main()
