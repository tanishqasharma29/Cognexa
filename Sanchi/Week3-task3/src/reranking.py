
import json
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parents[1]

DATASETS = {
    "A": BASE_DIR / "data/processed/dataset_A/chunks.json",
    "B": BASE_DIR / "data/processed/dataset_B/chunks.json"
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60
INITIAL_K = 10
FINAL_K = 5


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks(dataset):
    with open(DATASETS[dataset], "r", encoding="utf-8") as file:
        return json.load(file)


def dense_search(query, chunks, model, top_k=INITIAL_K):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
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


def bm25_search(query, chunks, top_k=INITIAL_K):
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenize(query))
    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "rank": rank,
            "score": float(scores[index]),
            "index": int(index)
        }
        for rank, index in enumerate(indices, start=1)
    ]


def rrf_fusion(dense_results, bm25_results):
    fused = {}

    for method, results in [
        ("dense", dense_results),
        ("bm25", bm25_results)
    ]:
        for result in results:
            index = result["index"]

            if index not in fused:
                fused[index] = {
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None
                }

            fused[index][f"{method}_rank"] = result["rank"]

            fused[index]["rrf_score"] += (
                1 / (RRF_K + result["rank"])
            )

    return sorted(
        fused.items(),
        key=lambda item: item[1]["rrf_score"],
        reverse=True
    )


def rerank(query, hybrid_results, chunks, reranker):
    candidates = hybrid_results[:INITIAL_K]

    pairs = [
        [query, chunks[index]["text"]]
        for index, data in candidates
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for (index, data), score in zip(candidates, scores):
        reranked.append({
            "index": index,
            "rerank_score": float(score),
            "rrf_score": data["rrf_score"],
            "dense_rank": data["dense_rank"],
            "bm25_rank": data["bm25_rank"]
        })

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked


def main():
    print("Available datasets: A and B")
    dataset = input("Select dataset (A/B): ").strip().upper()

    if dataset not in DATASETS:
        raise ValueError("Invalid dataset. Enter A or B.")

    chunks = load_chunks(dataset)

    print(f"\nDataset {dataset}: Loaded {len(chunks)} chunks")

    query = input("\nEnter your query: ").strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("\nRunning dense retrieval...")
    dense_results = dense_search(
        query,
        chunks,
        embedding_model
    )

    print("Running BM25...")
    bm25_results = bm25_search(
        query,
        chunks
    )

    print("Running hybrid RRF...")
    hybrid_results = rrf_fusion(
        dense_results,
        bm25_results
    )

    print("Running cross-encoder reranking...")
    reranked = rerank(
        query,
        hybrid_results,
        chunks,
        reranker
    )

    results = []

    for rank, result in enumerate(reranked[:FINAL_K], start=1):
        chunk = chunks[result["index"]]

        results.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "rerank_score": result["rerank_score"],
            "rrf_score": result["rrf_score"],
            "dense_rank": result["dense_rank"],
            "bm25_rank": result["bm25_rank"],
            "text": chunk["text"]
        })

    print("\n===== FINAL RERANKED RESULTS =====")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Rerank Score: {result['rerank_score']:.4f}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Document: {result['source_file']}")
        print(f"Page: {result['page']}")
        print(f"Dense Rank: {result['dense_rank']}")
        print(f"BM25 Rank: {result['bm25_rank']}")
        print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    main()
