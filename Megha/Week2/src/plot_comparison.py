
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

INPUT_FILE = Path("results/dataset_comparison.json")
OUTPUT_DIR = Path("results/graphs")

METHODS = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "hybrid_reranking": "Hybrid + Reranking"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Comparison results not found. Run py src\\compare_datasets.py first."
    )

with INPUT_FILE.open("r", encoding="utf-8") as f:
    comparison = json.load(f)

if "full_query_metrics" not in comparison:
    raise KeyError(
        "full_query_metrics not found. Run the corrected compare_datasets.py first."
    )

results = comparison["full_query_metrics"]


def save_graph(fig, filename):
    fig.savefig(
        OUTPUT_DIR / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        OUTPUT_DIR / f"{filename}.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)


def create_precision_recall_graph(method_key, method_name, metric):
    fig, ax = plt.subplots(figsize=(10, 6))

    if metric == "precision":
        labels = ["P@1", "P@3", "P@5", "P@10"]
    else:
        labels = ["R@1", "R@3", "R@5", "R@10"]

    values_a = [
        results[method_key][f"{metric}@{k}"]["dataset_a"]
        for k in [1, 3, 5, 10]
    ]

    values_b = [
        results[method_key][f"{metric}@{k}"]["dataset_b"]
        for k in [1, 3, 5, 10]
    ]

    x = np.arange(len(labels))
    width = 0.35

    bars_a = ax.bar(
        x - width / 2,
        values_a,
        width,
        label="Without OCR"
    )

    bars_b = ax.bar(
        x + width / 2,
        values_b,
        width,
        label="With OCR"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.bar_label(bars_a, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars_b, fmt="%.3f", padding=3, fontsize=9)

    ax.set_title(
        f"{method_name} - {metric.title()} Comparison (19 Queries)",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(f"{metric.title()} Score")
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()

    save_graph(
        fig,
        f"{method_key}_{metric}"
    )


def create_mrr_graph(method_key, method_name):
    fig, ax = plt.subplots(figsize=(9, 6))

    value_a = results[method_key]["mrr"]["dataset_a"]
    value_b = results[method_key]["mrr"]["dataset_b"]

    bars = ax.bar(
        ["Without OCR", "With OCR"],
        [value_a, value_b],
        width=0.45,
        label=method_name
    )

    ax.bar_label(bars, fmt="%.4f", padding=4)

    ax.set_title(
        f"{method_name} - MRR Comparison (19 Queries)",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel("Mean Reciprocal Rank")
    ax.set_ylim(0, 1.10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()

    save_graph(
        fig,
        f"{method_key}_mrr"
    )


for method_key, method_name in METHODS.items():
    create_precision_recall_graph(
        method_key, method_name, "precision"
    )

    create_precision_recall_graph(
        method_key, method_name, "recall"
    )

    create_mrr_graph(
        method_key, method_name
    )


print("\nSuccessfully generated all 12 graphs!")
print("Formats: PNG (300 DPI) and PDF")
print(f"Location: {OUTPUT_DIR.resolve()}")
print(f"Total files: {len(list(OUTPUT_DIR.glob('*.png'))) + len(list(OUTPUT_DIR.glob('*.pdf')))}")
