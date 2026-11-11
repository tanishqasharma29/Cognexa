
import csv
import json
from pathlib import Path

A_FILE = Path("results/benchmark_results.json")
B_FILE = Path("results/benchmark_results_dataset_b.json")

JSON_OUTPUT = Path("results/dataset_comparison.json")
CSV_OUTPUT = Path("results/final_comparison.csv")

METHODS = [
    "bm25",
    "dense",
    "hybrid",
    "hybrid_reranking"
]

K_VALUES = [1, 3, 5, 10]

QUALITY_METRICS = (
    [f"precision@{k}" for k in K_VALUES]
    + [f"recall@{k}" for k in K_VALUES]
    + ["mrr"]
)

METRICS = QUALITY_METRICS + ["retrieval_time_seconds"]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def average(values):
    return sum(values) / len(values) if values else None


def main():
    a = load_json(A_FILE)
    b = load_json(B_FILE)

    a_queries = {q["query_id"]: q for q in a["queries"]}
    b_queries = {q["query_id"]: q for q in b["queries"]}

    if set(a_queries) != set(b_queries):
        raise ValueError("Dataset A and B do not have identical query IDs.")

    query_ids = sorted(a_queries)

    common_ids = [
        qid for qid in query_ids
        if a_queries[qid]["answerable"]
        and b_queries[qid]["answerable"]
    ]

    ocr_ids = [
        qid for qid in query_ids
        if not a_queries[qid]["answerable"]
        and b_queries[qid]["answerable"]
    ]

    comparison = {
        "dataset_a_chunks": a["num_chunks"],
        "dataset_b_chunks": b["num_chunks"],
        "total_query_count": len(query_ids),
        "common_query_count": len(common_ids),
        "ocr_specific_query_count": len(ocr_ids),
        "full_query_metrics": {},
        "common_query_metrics": {},
        "ocr_specific_metrics": {}
    }

    rows = []

    for method in METHODS:

        comparison["full_query_metrics"][method] = {}
        comparison["common_query_metrics"][method] = {}
        comparison["ocr_specific_metrics"][method] = {}

        for metric in METRICS:

            # FULL FIXED QUERY SET: 19 queries

            values_a = []
            values_b = []

            for qid in query_ids:
                qa = a_queries[qid]
                qb = b_queries[qid]

                if qa["answerable"]:
                    values_a.append(
                        qa["methods"][method]["metrics"][metric]
                    )
                elif metric in QUALITY_METRICS:
                    # OCR evidence is absent in Dataset A.
                    values_a.append(0.0)
                else:
                    # Dataset A did not run retrieval for OCR queries.
                    # Do not treat missing retrieval time as zero.
                    values_a.append(None)

                values_b.append(
                    qb["methods"][method]["metrics"][metric]
                )

            # Average A's retrieval time only over queries actually run.
            if metric == "retrieval_time_seconds":
                value_a = average([
                    a_queries[qid]["methods"][method]["metrics"][metric]
                    for qid in common_ids
                ])
            else:
                value_a = average(values_a)

            value_b = average(values_b)

            change = value_b - value_a

            comparison["full_query_metrics"][method][metric] = {
                "dataset_a": value_a,
                "dataset_b": value_b,
                "change": change
            }

            rows.append({
                "evaluation": "full_fixed_query_set",
                "method": method,
                "metric": metric,
                "dataset_a": value_a,
                "dataset_b": value_b,
                "change": change,
                "query_count": len(query_ids)
            })

            # COMMON QUERIES

            common_a = [
                a_queries[qid]["methods"][method]["metrics"][metric]
                for qid in common_ids
            ]

            common_b = [
                b_queries[qid]["methods"][method]["metrics"][metric]
                for qid in common_ids
            ]

            ca = average(common_a)
            cb = average(common_b)

            comparison["common_query_metrics"][method][metric] = {
                "dataset_a": ca,
                "dataset_b": cb,
                "change": cb - ca
            }

            rows.append({
                "evaluation": "common_queries",
                "method": method,
                "metric": metric,
                "dataset_a": ca,
                "dataset_b": cb,
                "change": cb - ca,
                "query_count": len(common_ids)
            })

            # OCR-SPECIFIC QUERIES: DATASET B

            ocr_values = [
                b_queries[qid]["methods"][method]["metrics"][metric]
                for qid in ocr_ids
            ]

            ov = average(ocr_values)

            comparison["ocr_specific_metrics"][method][metric] = ov

            rows.append({
                "evaluation": "ocr_specific_queries",
                "method": method,
                "metric": metric,
                "dataset_a": None,
                "dataset_b": ov,
                "change": None,
                "query_count": len(ocr_ids)
            })

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with JSON_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with CSV_OUTPUT.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "evaluation",
                "method",
                "metric",
                "dataset_a",
                "dataset_b",
                "change",
                "query_count"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== DATASET COMPARISON =====")
    print(f"Dataset A chunks: {a['num_chunks']}")
    print(f"Dataset B chunks: {b['num_chunks']}")
    print(f"Total fixed queries: {len(query_ids)}")
    print(f"Common queries: {len(common_ids)}")
    print(f"OCR-specific queries: {len(ocr_ids)}")

    print("\n===== FULL QUERY SET RESULTS =====")

    for method in METHODS:
        print(f"\n{method.upper()}")

        for metric in METRICS:
            result = comparison["full_query_metrics"][method][metric]

            av = result["dataset_a"]
            bv = result["dataset_b"]

            av = "N/A" if av is None else f"{av:.4f}"
            bv = "N/A" if bv is None else f"{bv:.4f}"

            change = result["change"]
            change = "N/A" if change is None else f"{change:+.4f}"

            print(f"{metric:24} {av:>9} -> {bv:>9}  Change: {change}")

    print("\n===== SAVED FILES =====")
    print(JSON_OUTPUT)
    print(CSV_OUTPUT)


if __name__ == "__main__":
    main()
