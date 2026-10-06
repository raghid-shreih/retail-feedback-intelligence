"""Generate aggregate evaluation figures from a local recommendation JSON report.

Run from the repository root after `pip install -e '.[plots]'`:
    python scripts/build_gallery.py results/recommendations.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def summarize_report(report: dict) -> dict:
    """Recompute figures from predictions, rejecting inconsistent reports."""
    rows = report["rows"]
    if not rows:
        raise ValueError("At least one attempted prediction is required")
    matrix = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    actuals = []
    for row in rows:
        actual, predicted = row["actual"], row["predicted"]
        if type(actual) is not int or actual not in (0, 1):
            raise ValueError("Actual labels must be 0 or 1")
        if predicted is not None and (type(predicted) is not int or predicted not in (0, 1)):
            raise ValueError("Predictions must be 0, 1, or null")
        actuals.append(actual)
        if predicted is not None:
            matrix[{(0, 0): "tn", (0, 1): "fp", (1, 0): "fn", (1, 1): "tp"}[(actual, predicted)]] += 1
    attempted = len(rows)
    valid = sum(matrix.values())
    correct = matrix["tn"] + matrix["tp"]
    if report["attempted"] != attempted or report["valid"] != valid or report["confusion_matrix_valid_only"] != matrix:
        raise ValueError("Report totals do not match row-level predictions")
    if abs(report["accuracy_all_attempts"] - correct / attempted) > 1e-8:
        raise ValueError("Reported accuracy does not match row-level predictions")
    majority = max(actuals.count(0), actuals.count(1))
    return {"matrix": matrix, "attempted": attempted, "valid": valid, "correct": correct, "majority": majority}


def save_gallery(report: dict, directory: str | Path) -> list[Path]:
    summary = summarize_report(report)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    burgundy, rose, ink, grid = "#672744", "#BD7898", "#24303B", "#EDF0F2"
    paths = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": ink, "axes.labelcolor": ink,
                         "svg.fonttype": "none", "svg.hashsalt": "retail-evaluation-gallery"}):
        n, correct, majority = summary["attempted"], summary["correct"], summary["majority"]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        bars = ax.barh(["Majority-class baseline", "Recommendation prompt"],
                       [majority / n, correct / n], color=[rose, burgundy], height=.55)
        for bar, count in zip(bars, [majority, correct]):
            ax.text(bar.get_width() + .015, bar.get_y() + bar.get_height() / 2,
                    f"{count}/{n}  ({count / n:.0%})", va="center", color=ink, weight="bold")
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, .25, .5, .75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Correct predictions / all attempts")
        ax.set_title(f"Accuracy on one {n}-review sample", loc="left", weight="bold", pad=16)
        ax.grid(axis="x", color=grid)
        ax.set_axisbelow(True)
        fig.tight_layout()
        paths.append(destination / "sample-accuracy.svg")
        fig.savefig(paths[-1], format="svg", metadata={"Date": None})
        plt.close(fig)

        m = summary["matrix"]
        values = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        ax.imshow(values, cmap=ListedColormap(["#F3E8ED", "#D8AEC1", burgundy]), vmin=0, vmax=max(map(max, values)) or 1)
        ax.set_xticks([0, 1], labels=["No (0)", "Yes (1)"])
        ax.set_yticks([0, 1], labels=["No (0)", "Yes (1)"])
        ax.set_xlabel("Predicted recommendation")
        ax.set_ylabel("Dataset label")
        ax.set_title(f"Valid predictions: {summary['valid']} of {n}", loc="left", weight="bold", pad=16)
        for i in range(2):
            for j in range(2):
                value = values[i][j]
                ax.text(j, i, str(value), ha="center", va="center", weight="bold", fontsize=23,
                        color="white" if value >= max(map(max, values)) * .65 and value else ink)
        ax.set_xticks([-.5, .5, 1.5], minor=True)
        ax.set_yticks([-.5, .5, 1.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=4)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.tight_layout()
        paths.append(destination / "confusion-matrix.svg")
        fig.savefig(paths[-1], format="svg", metadata={"Date": None})
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Local recommendation JSON produced by retail-feedback recommend")
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    for path in save_gallery(report, args.output):
        print(path)


if __name__ == "__main__":
    main()
