
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parents[1]

DATASETS = {
    "A": BASE_DIR / "data/processed/dataset_A_actual/chunks.json",
    "B": BASE_DIR / "data/processed/dataset_B_actual/chunks.json"
}
TOP_K = 5


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks(dataset):
    with open(DATASETS[dataset], "r", encoding="utf-8") as file:
        return json.load(file)


def build_bm25(chunks):
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]
    return BM25Okapi(tokenized_chunks)


def search(query, chunks, bm25, top_k=TOP_K):
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]

        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "text": chunk["text"]
        })

    return results


def main():
    print("Available datasets: A and B")
    dataset = input("Select dataset (A/B): ").strip().upper()

    if dataset not in DATASETS:
        raise ValueError("Invalid dataset. Enter A or B.")

    chunks = load_chunks(dataset)

    print(f"\nDataset {dataset}: Loaded {len(chunks)} chunks")

    bm25 = build_bm25(chunks)

    print("BM25 index created.")

    query = input("\nEnter your query: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    results = search(query, chunks, bm25, TOP_K)

    print("\n===== BM25 LEXICAL RETRIEVAL RESULTS =====")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Document: {result['source_file']}")
        print(f"Page: {result['page']}")
        print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    main()
