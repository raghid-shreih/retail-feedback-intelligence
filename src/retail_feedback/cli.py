"""Command line interface for bounded, reproducible experiments."""

import argparse
import json
from pathlib import Path

from .client import ModelAPIError, OpenAIClient
from .data import load_reviews, sample_reviews
from .evaluation import compare, evaluate_recommendations
from .prompts import ANALYSIS_PROMPTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retail-feedback")
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summary", help="Inspect a local CSV without an API call")
    summary.add_argument("csv", type=Path)
    for name in ("compare", "recommend"):
        command = commands.add_parser(name)
        command.add_argument("csv", type=Path)
        command.add_argument("--model", required=True, help="OpenAI model ID; no default is assumed")
        command.add_argument("--limit", type=int, default=5, help="Reviews sampled; default 5")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--output", type=Path, help="Write JSON report locally")
        if name == "compare":
            command.add_argument("--variants", nargs="+", choices=sorted(ANALYSIS_PROMPTS),
                                 default=list(ANALYSIS_PROMPTS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        frame = load_reviews(args.csv)
        if args.command == "summary":
            report = {"rows_with_review": len(frame), "columns": list(frame.columns),
                      "recommendation_labels": frame["recommended_ind"].value_counts(dropna=False).to_dict()
                      if "recommended_ind" in frame else None}
        else:
            sampled = sample_reviews(frame, args.limit, args.seed)
            client = OpenAIClient(args.model)
            report = compare(sampled, client.complete, args.variants) if args.command == "compare" else \
                evaluate_recommendations(sampled, client.complete)
            report = {"model": args.model, "seed": args.seed, "sample_size": len(sampled), **report}
        rendered = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        if getattr(args, "output", None):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(f"Saved {args.output}")
        else:
            print(rendered)
        return 0
    except (FileNotFoundError, ValueError, KeyError, ModelAPIError) as exc:
        build_parser().exit(2, f"error: {exc}\n")
