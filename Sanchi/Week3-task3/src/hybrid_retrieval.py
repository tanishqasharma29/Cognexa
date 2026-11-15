
import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]

DATASETS = {
    "A": BASE_DIR / "data/processed/dataset_A_actual/chunks.json",
    "B": BASE_DIR / "data/processed/dataset_B_actual/chunks.json"
}

MODEL_NAME = "all-MiniLM-L6-v2"
CANDIDATE_K = 10
TOP_K = 5
RRF_K = 60


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks(dataset):
    with open(DATASETS[dataset], "r", encoding="utf-8") as file:
        return json.load(file)


def dense_search(query, chunks, model, top_k=CANDIDATE_K):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "rank": rank,
            "score": float(scores[index]),
            "index": int(index)
        }
        for rank, index in enumerate(indices, start=1)
    ]


def bm25_search(query, chunks, top_k=CANDIDATE_K):
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "rank": rank,
            "score": float(scores[index]),
            "index": int(index)
        }
        for rank, index in enumerate(indices, start=1)
    ]


def reciprocal_rank_fusion(dense_results, bm25_results):
    fused = {}

    for method, results in [
        ("dense", dense_results),
        ("bm25", bm25_results)
    ]:
        for result in results:
            index = result["index"]

            if index not in fused:
                fused[index] = {
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0
                }

            fused[index][f"{method}_rank"] = result["rank"]
            fused[index][f"{method}_score"] = result["score"]

            fused[index]["rrf_score"] += (
                1 / (RRF_K + result["rank"])
            )

    return fused


def main():
    print("Available datasets: A and B")
    dataset = input("Select dataset (A/B): ").strip().upper()

    if dataset not in DATASETS:
        raise ValueError("Invalid dataset. Enter A or B.")

    chunks = load_chunks(dataset)

    print(f"\nDataset {dataset}: Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)

    query = input("\nEnter your query: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    dense_results = dense_search(
        query,
        chunks,
        model
    )

    bm25_results = bm25_search(
        query,
        chunks
    )

    fused = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    ranked = sorted(
        fused.items(),
        key=lambda item: item[1]["rrf_score"],
        reverse=True
    )

    results = []

    for rank, (index, data) in enumerate(ranked[:TOP_K], start=1):
        chunk = chunks[index]

        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "rrf_score": data["rrf_score"],
            "dense_rank": data["dense_rank"],
            "dense_score": data["dense_score"],
            "bm25_rank": data["bm25_rank"],
            "bm25_score": data["bm25_score"],
            "text": chunk["text"]
        })

    print("\n===== HYBRID RRF RESULTS =====")

    for result in results:
        print(f"\nHybrid Rank: {result['rank']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Document: {result['source_file']}")
        print(f"Page: {result['page']}")
        print(f"Dense Rank: {result['dense_rank']}")
        print(f"Dense Score: {result['dense_score']}")
        print(f"BM25 Rank: {result['bm25_rank']}")
        print(f"BM25 Score: {result['bm25_score']}")
        print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    main()
