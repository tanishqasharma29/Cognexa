import json

with open("results/benchmark_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

methods = [
    "lexical",
    "dense",
    "hybrid",
    "hybrid_reranking"
]

answerable = [
    q for q in data["queries"]
    if q["answerable"]
]

print("\nMETHOD COMPARISON")
print("-" * 70)
print(f"{'Method':<24}{'Hit@5':<12}{'MRR':<12}{'Avg Time (s)':<15}")
print("-" * 70)

for method in methods:

    hits = [
        q["methods"][method]["hit_at_5"]
        for q in answerable
    ]

    mrrs = [
        q["methods"][method]["mrr"]
        for q in answerable
    ]

    times = [
        q["methods"][method]["time_seconds"]
        for q in data["queries"]
    ]

    avg_hit = sum(hits) / len(hits)
    avg_mrr = sum(mrrs) / len(mrrs)
    avg_time = sum(times) / len(times)

    print(
        f"{method:<24}"
        f"{avg_hit:<12.3f}"
        f"{avg_mrr:<12.3f}"
        f"{avg_time:<15.4f}"
    )

print("\nUNSUPPORTED QUERY")
print("-" * 70)

for q in data["queries"]:

    if not q["answerable"]:

        print(f"Query: {q['query']}")

        for method in methods:

            clean = q["methods"][method]["unsupported_clean"]

            print(
                f"  {method:<22}: "
                f"{'No evidence returned' if clean else 'Returned candidates'}"
            )
