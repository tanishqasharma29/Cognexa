
import json
import re
from pathlib import Path

A_FILE = Path("data/processed/chunks.json")
B_FILE = Path("data/processed/chunks_dataset_b.json")
OUTPUT_FILE = Path("results/benchmark_queries.json")

NUM_SELECTABLE_QUERIES = 10
NUM_OCR_QUERIES = 10

STOP_WORDS = {
    "this", "that", "with", "from", "have", "were", "will",
    "would", "could", "should", "their", "there", "these",
    "those", "which", "when", "where", "what", "about",
    "into", "also", "than", "then", "them", "they", "your",
    "some", "such", "only", "more", "most", "very", "each",
    "being", "been", "does", "document", "page", "using",
    "used", "between", "through", "after", "before"
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_evenly(items, count):
    if not items:
        return []

    if len(items) <= count:
        return items

    if count == 1:
        return [items[len(items) // 2]]

    indexes = [
        round(i * (len(items) - 1) / (count - 1))
        for i in range(count)
    ]

    return [items[i] for i in indexes]


def make_query(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", text.lower())

    selected = []
    seen = set()

    for word in words:
        if word not in STOP_WORDS and word not in seen:
            selected.append(word)
            seen.add(word)

        if len(selected) == 8:
            break

    if len(selected) < 4:
        selected = words[:8]

    if not selected:
        return None

    return "Find information about " + " ".join(selected)


def main():
    if not A_FILE.exists() or not B_FILE.exists():
        raise FileNotFoundError(
            "Run extraction and both chunking scripts first."
        )

    chunks_a = load_json(A_FILE)
    chunks_b = load_json(B_FILE)

    ids_a = {c["chunk_id"] for c in chunks_a}
    ids_b = {c["chunk_id"] for c in chunks_b}

    # Select chunks that exist in both datasets.
    selectable = [
        c for c in chunks_b
        if c["extraction_method"] == "text"
        and c["chunk_id"] in ids_a
        and len(c["text"].strip()) >= 100
    ]

    # Select OCR-only chunks from Dataset B.
    ocr_chunks = [
        c for c in chunks_b
        if c["extraction_method"] == "ocr"
        and len(c["text"].strip()) >= 100
    ]

    selected_a = sample_evenly(
        selectable, NUM_SELECTABLE_QUERIES
    )

    selected_ocr = sample_evenly(
        ocr_chunks, NUM_OCR_QUERIES
    )

    queries = []

    for i, chunk in enumerate(selected_a, start=1):
        query_text = make_query(chunk["text"])

        if not query_text:
            continue

        queries.append({
            "query_id": f"qA{i}",
            "query": query_text,
            "expected_chunks_a": [chunk["chunk_id"]],
            "expected_chunks_b": [chunk["chunk_id"]]
        })

    for i, chunk in enumerate(selected_ocr, start=1):
        query_text = make_query(chunk["text"])

        if not query_text:
            continue

        queries.append({
            "query_id": f"qOCR{i}",
            "query": query_text,
            "expected_chunks_a": [],
            "expected_chunks_b": [chunk["chunk_id"]]
        })

    if not queries:
        raise ValueError("No benchmark queries could be generated.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)

    print("\nBenchmark query generation complete.")
    print(f"Selectable queries: {len(selected_a)}")
    print(f"OCR queries: {len(selected_ocr)}")
    print(f"Total queries generated: {len(queries)}")
    print(f"Saved: {OUTPUT_FILE}")

    print("\nSample queries:")

    for query in queries[:5]:
        print(f"\n{query['query_id']}: {query['query']}")
        print(f"Dataset A relevant IDs: {query['expected_chunks_a']}")
        print(f"Dataset B relevant IDs: {query['expected_chunks_b']}")


if __name__ == "__main__":
    main()
