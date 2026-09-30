"""Load both the source Kaggle CSV and semicolon-separated notebook exports."""

import re
from pathlib import Path

import pandas as pd


def load_reviews(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [re.sub(r"[^a-z0-9]+", "_", str(col).lower()).strip("_") for col in frame.columns]
    frame = frame.loc[:, ~frame.columns.str.match(r"^unnamed(_\d+)?$")]
    if "review_text" not in frame:
        raise ValueError("CSV needs a Review Text (or Review.Text) column")
    frame["review_text"] = frame["review_text"].astype("string").str.strip()
    frame = frame.dropna(subset=["review_text"])
    frame = frame[frame["review_text"] != ""].copy()
    return frame


def sample_reviews(frame: pd.DataFrame, limit: int, seed: int = 42) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("--limit must be at least 1")
    return frame.sample(n=min(limit, len(frame)), random_state=seed)


def recommendation_label(value: object) -> int:
    if pd.isna(value) or str(value).strip() not in {"0", "1", "0.0", "1.0"}:
        raise ValueError(f"Expected recommendation label 0 or 1, got {value!r}")
    return int(float(value))
