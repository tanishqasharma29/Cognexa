
import json
from pathlib import Path

INPUT_FILE = Path("results/benchmark_results.json")
OUTPUT_FILE = Path("results/benchmark_summary.json")

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
    "precision_at_1",
    "precision_at_3",
    "precision_at_5",
    "mrr"
]


def average(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    answerable = [q for q in queries if q["answerable"]]

    summary = {
        "total_queries": len(queries),
        "answerable_queries": len(answerable),
        "unsupported_queries": len(queries) - len(answerable),
        "methods": {}
    }

    print("\nRECALLROOM - RETRIEVAL BENCHMARK")
    print("=" * 105)
    print(f"Total queries: {len(queries)}")
    print(f"Answerable: {len(answerable)}")
    print(f"Unsupported: {len(queries) - len(answerable)}")

    for method in METHODS:
        method_summary = {}

        for metric in METRICS:
            values = [
                q["methods"][method][metric]
                for q in answerable
            ]
            method_summary[metric] = average(values)

        times = [
            q["methods"][method]["time_seconds"]
            for q in queries
        ]

        method_summary["avg_time_seconds"] = average(times)
        summary["methods"][method] = method_summary

    print("\nMETHOD COMPARISON")
    print("-" * 105)
    print(
        f"{'Method':<22}"
        f"{'Hit@5':>9}"
        f"{'R@1':>9}"
        f"{'R@3':>9}"
        f"{'R@5':>9}"
        f"{'P@1':>9}"
        f"{'P@3':>9}"
        f"{'P@5':>9}"
        f"{'MRR':>9}"
        f"{'Time(s)':>11}"
    )

    for method, values in summary["methods"].items():
        print(
            f"{method:<22}"
            f"{values['hit_at_5']:>9.3f}"
            f"{values['recall_at_1']:>9.3f}"
            f"{values['recall_at_3']:>9.3f}"
            f"{values['recall_at_5']:>9.3f}"
            f"{values['precision_at_1']:>9.3f}"
            f"{values['precision_at_3']:>9.3f}"
            f"{values['precision_at_5']:>9.3f}"
            f"{values['mrr']:>9.3f}"
            f"{values['avg_time_seconds']:>11.4f}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
