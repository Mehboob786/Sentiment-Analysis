from pathlib import Path

import pandas as pd
from datasets import load_dataset


LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def _convert_split(split_df: pd.DataFrame) -> pd.DataFrame:
    out = split_df[["text", "label"]].copy()
    out["label"] = out["label"].map(LABEL_MAP)
    return out


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    train_df = _convert_split(pd.DataFrame(ds["train"]))
    val_df = _convert_split(pd.DataFrame(ds["validation"]))
    test_df = _convert_split(pd.DataFrame(ds["test"]))

    train_df.to_csv(data_dir / "train.csv", index=False)
    val_df.to_csv(data_dir / "val.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    print("Saved:")
    print(f"- {data_dir / 'train.csv'} ({len(train_df)} rows)")
    print(f"- {data_dir / 'val.csv'} ({len(val_df)} rows)")
    print(f"- {data_dir / 'test.csv'} ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
