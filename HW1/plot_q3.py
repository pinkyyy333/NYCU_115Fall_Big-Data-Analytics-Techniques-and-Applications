"""Plot Q3 accuracy from the completed rank-16 experiments (no training).

Requires matplotlib. Run from any working directory: python plot_q3.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "benchmark_results.csv"
OUTPUT = ROOT / "figures"
COMPONENTS = ["default", "25%", "50%", "bottom"]
LABELS = ["Top 16", "25% position", "50% position", "Bottom 16"]
COLORS = ["#2563A6", "#3A8692", "#8B72A5", "#C17A45"]


def main():
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = {}
    for component in COMPONENTS:
        matches = [
            row for row in rows
            if row["method"] == "pissa"
            and row["status"] == "ok"
            and float(row["rank"]) == 16
            and row["component"] == component
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one successful rank-16 run: {component}")
        selected[component] = matches[0]

    OUTPUT.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })

    def draw(ax, task, title):
        values = [100 * float(selected[c][f"{task}_accuracy"]) for c in COMPONENTS]
        totals = {int(selected[c][f"{task}_total"]) for c in COMPONENTS}
        if len(totals) != 1:
            raise ValueError("Evaluation sizes differ across components")
        bars = ax.bar(LABELS, values, color=COLORS, width=0.62)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=5)
        ax.set_title(f"{title} — Rank 16", fontsize=13, fontweight="bold", pad=13)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Selected singular-value range", labelpad=10)
        ax.set_ylim(0, max(values) * 1.25)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        return totals.pop()

    for task, title in [("gsm8k", "GSM8K"), ("math", "MATH")]:
        fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
        draw(ax, task, title)
        fig.savefig(OUTPUT / f"q3_{task}_accuracy.png", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9), constrained_layout=True)
    for ax, task, title in zip(axes, ["gsm8k", "math"], ["GSM8K", "MATH"]):
        draw(ax, task, title)
    fig.savefig(OUTPUT / "q3_accuracy_comparison.png", dpi=300)
    fig.savefig(OUTPUT / "q3_accuracy_comparison.pdf")
    plt.close(fig)
    for task in ["gsm8k", "math"]:
        print(task, {c: float(selected[c][f"{task}_accuracy"]) * 100 for c in COMPONENTS})
    print(f"Saved plots to {OUTPUT}")


if __name__ == "__main__":
    main()
