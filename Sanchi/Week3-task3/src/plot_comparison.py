import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

INPUT = Path("results/dataset_comparison.json")
OUTPUT = Path("results/graphs")
OUTPUT.mkdir(parents=True, exist_ok=True)

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset_a = data["dataset_a"]["metrics"]
dataset_b = data["dataset_b"]["metrics"]

methods = ["lexical", "dense", "hybrid", "hybrid_reranking"]

labels = {
    "lexical": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "hybrid_reranking": "Hybrid + Reranking"
}

metrics = [
    "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
    "precision_at_1", "precision_at_3", "precision_at_5",
    "precision_at_10", "mrr", "avg_time_seconds"
]

for metric in metrics:
    a_values = [dataset_a[m][metric] for m in methods]
    b_values = [dataset_b[m][metric] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, a_values, width, label="Dataset A")
    ax.bar(x + width/2, b_values, width, label="Dataset B")

    ax.set_title(metric.replace("_", " ").upper())
    ax.set_ylabel("Seconds" if metric == "avg_time_seconds" else "Score")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in methods], rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    filename = OUTPUT / f"{metric}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)

    print(f"Saved: {filename}")

print("\nAll comparison graphs generated successfully.")
