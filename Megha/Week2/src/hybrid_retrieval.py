import json
import re
import numpy as np
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.json")

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5
RRF_K = 60


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def dense_search(query, chunks, model, top_k=TOP_K):

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

    results = []

    for rank, index in enumerate(indices, start=1):
        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "index": int(index)
        })

    return results


def bm25_search(query, chunks, top_k=TOP_K):

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index in enumerate(indices, start=1):
        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "index": int(index)
        })

    return results


def reciprocal_rank_fusion(dense_results, bm25_results):

    fused = {}

    for result in dense_results:

        index = result["index"]

        if index not in fused:
            fused[index] = {
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0
            }

        fused[index]["dense_rank"] = result["rank"]
        fused[index]["dense_score"] = result["score"]

        fused[index]["rrf_score"] += (
            1 / (RRF_K + result["rank"])
        )

    for result in bm25_results:

        index = result["index"]

        if index not in fused:
            fused[index] = {
                "dense_rank": None,
                "dense_score": None,
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": 0.0
            }

        fused[index]["bm25_rank"] = result["rank"]
        fused[index]["bm25_score"] = result["score"]

        fused[index]["rrf_score"] += (
            1 / (RRF_K + result["rank"])
        )

    return fused


def main():

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)

    query = input("\nEnter your query: ")

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
        key=lambda x: x[1]["rrf_score"],
        reverse=True
    )

    print("\n===== HYBRID RRF RESULTS =====")

    for rank, (index, data) in enumerate(
        ranked[:TOP_K],
        start=1
    ):

        chunk = chunks[index]

        print(f"\nHybrid Rank: {rank}")
        print(f"RRF Score: {data['rrf_score']:.6f}")
        print(f"Chunk: {chunk['chunk_id']}")
        print(f"Document: {chunk['source_file']}")
        print(f"Page: {chunk['page']}")

        print(f"Dense Rank: {data['dense_rank']}")
        print(f"Dense Score: {data['dense_score']}")

        print(f"BM25 Rank: {data['bm25_rank']}")
        print(f"BM25 Score: {data['bm25_score']}")

        print(f"Text: {chunk['text'][:500]}...")


if __name__ == "__main__":
    main()