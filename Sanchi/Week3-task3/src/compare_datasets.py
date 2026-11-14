import json
import csv
from pathlib import Path

A_FILE = Path("results/benchmark_results.json")
B_FILE = Path("results/benchmark_results_dataset_b.json")
OUTPUT_FILE = Path("results/dataset_comparison.json")
CSV_FILE = Path("results/dataset_comparison.csv")

METHODS = [
    "lexical",
    "dense",
    "hybrid",
    "hybrid_reranking"
]

METRICS = [
    "hit_at_5",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "precision_at_1",
    "precision_at_3",
    "precision_at_5",
    "precision_at_10",
    "mrr",
    "avg_time_seconds"
]


def average(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def calculate_metrics(data):
    answerable = [
        q for q in data["queries"]
        if q["answerable"]
    ]

    summary = {}

    for method in METHODS:
        summary[method] = {}

        for metric in METRICS:
            if metric == "avg_time_seconds":
                values = [
                    q["methods"][method]["time_seconds"]
                    for q in data["queries"]
                ]
            else:
                values = [
                    q["methods"][method].get(metric)
                    for q in answerable
                ]

            summary[method][metric] = average(values)

    return summary


def main():
    with open(A_FILE, "r", encoding="utf-8") as f:
        dataset_a = json.load(f)

    with open(B_FILE, "r", encoding="utf-8") as f:
        dataset_b = json.load(f)

    if [q["query_id"] for q in dataset_a["queries"]] != [
        q["query_id"] for q in dataset_b["queries"]
    ]:
        raise ValueError("Dataset A and B query sets do not match.")

    a = calculate_metrics(dataset_a)
    b = calculate_metrics(dataset_b)

    comparison = {
        "dataset_a": {
            "num_queries": dataset_a["benchmark"]["num_queries"],
            "num_chunks": 9,
            "metrics": a
        },
        "dataset_b": {
            "num_queries": dataset_b["benchmark"]["num_queries"],
            "num_chunks": dataset_b["benchmark"]["num_chunks"],
            "metrics": b
        },
        "difference_b_minus_a": {}
    }

    for method in METHODS:
        comparison["difference_b_minus_a"][method] = {}

        for metric in METRICS:
            av = a[method][metric]
            bv = b[method][metric]

            comparison["difference_b_minus_a"][method][metric] = (
                bv - av if av is not None and bv is not None else None
            )

    print("\nDATASET A vs DATASET B")
    print("=" * 100)

    for metric in METRICS:
        print(f"\n{metric.upper()}")
        print("-" * 75)
        print(
            f"{'Method':<22}"
            f"{'Dataset A':>13}"
            f"{'Dataset B':>13}"
            f"{'Change':>13}"
        )

        for method in METHODS:
            av = a[method][metric]
            bv = b[method][metric]
            change = (
                bv - av if av is not None and bv is not None else None
            )

            print(
                f"{method:<22}"
                f"{av:>13.4f}"
                f"{bv:>13.4f}"
                f"{change:>+13.4f}"
                if change is not None
                else f"{method:<22}{'N/A':>13}{'N/A':>13}{'N/A':>13}"
            )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method", "Metric", "Dataset A",
            "Dataset B", "Change (B-A)"
        ])

        for method in METHODS:
            for metric in METRICS:
                av = a[method][metric]
                bv = b[method][metric]
                change = (
                    bv - av if av is not None and bv is not None else None
                )
                writer.writerow([method, metric, av, bv, change])

    print(f"\nSaved JSON: {OUTPUT_FILE}")
    print(f"Saved CSV: {CSV_FILE}")


if __name__ == "__main__":
    main()
