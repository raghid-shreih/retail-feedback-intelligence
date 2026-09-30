"""Validation and evaluation, with invalid outputs counted as failures."""

import json
from collections.abc import Callable

import pandas as pd

from .data import recommendation_label
from .prompts import ANALYSIS_PROMPTS, recommendation_prompt

FIELDS = {"Category", "Sentiment", "Summary", "Personalized_Message", "Retail_Insight"}


def parse_analysis(text: str) -> dict:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("Analysis response needs exactly the five documented keys")
    if any(not isinstance(v, str) or not v.strip() for v in value.values()):
        raise ValueError("Analysis fields must be nonempty strings")
    if value["Sentiment"] not in {"Positive", "Negative", "Neutral"}:
        raise ValueError("Invalid sentiment")
    return value


def parse_recommendation(text: str) -> int:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"Recommended_IND", "Reason"}:
        raise ValueError("Recommendation response needs Recommended_IND and Reason")
    if not isinstance(value["Reason"], str) or not value["Reason"].strip():
        raise ValueError("Reason must be a nonempty string")
    label = value["Recommended_IND"]
    if type(label) is not int or label not in (0, 1):
        raise ValueError("Recommended_IND must be integer 0 or 1")
    return label


def compare(frame: pd.DataFrame, complete: Callable[[str], str], variants: list[str]) -> dict:
    unknown = set(variants) - set(ANALYSIS_PROMPTS)
    if unknown:
        raise ValueError(f"Unknown prompt variants: {sorted(unknown)}")
    rows = []
    for index, record in frame.iterrows():
        for variant in variants:
            row = {"source_row": int(index), "variant": variant}
            try:
                row["analysis"] = parse_analysis(complete(ANALYSIS_PROMPTS[variant](record["review_text"])))
                row["valid"] = True
            except (ValueError, json.JSONDecodeError) as exc:
                row.update(valid=False, error=str(exc))
            rows.append(row)
    summary = {v: {"attempted": len(frame), "valid": sum(r["valid"] for r in rows if r["variant"] == v)} for v in variants}
    return {"summary": summary, "rows": rows}


def evaluate_recommendations(frame: pd.DataFrame, complete: Callable[[str], str]) -> dict:
    if "recommended_ind" not in frame:
        raise ValueError("CSV needs a Recommended IND (or Recommended.IND) column")
    rows = []
    matrix = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    correct = 0
    for index, record in frame.iterrows():
        actual = recommendation_label(record["recommended_ind"])
        row = {"source_row": int(index), "actual": actual}
        try:
            prediction = parse_recommendation(complete(recommendation_prompt(record["review_text"])))
            row["predicted"] = prediction
            correct += prediction == actual
            matrix[{(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}[(actual, prediction)]] += 1
        except (ValueError, json.JSONDecodeError) as exc:
            row.update(predicted=None, error=str(exc))
        rows.append(row)
    total = len(rows)
    valid = sum(r["predicted"] is not None for r in rows)
    return {"attempted": total, "valid": valid, "accuracy_all_attempts": correct / total if total else None,
            "accuracy_valid_only": correct / valid if valid else None, "confusion_matrix_valid_only": matrix, "rows": rows}
