import json
import re
import numpy as np
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


CHUNKS_FILE = Path("data/processed/chunks.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60
INITIAL_K = 10
FINAL_K = 5


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


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

    scores = bm25.get_scores(
        tokenize(query)
    )

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

    for result in dense_results:

        index = result["index"]

        if index not in fused:
            fused[index] = {
                "rrf_score": 0,
                "dense_rank": None,
                "bm25_rank": None
            }

        fused[index]["dense_rank"] = result["rank"]

        fused[index]["rrf_score"] += (
            1 / (RRF_K + result["rank"])
        )

    for result in bm25_results:

        index = result["index"]

        if index not in fused:
            fused[index] = {
                "rrf_score": 0,
                "dense_rank": None,
                "bm25_rank": None
            }

        fused[index]["bm25_rank"] = result["rank"]

        fused[index]["rrf_score"] += (
            1 / (RRF_K + result["rank"])
        )

    return sorted(
        fused.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True
    )


def rerank(query, hybrid_results, chunks, reranker):

    candidates = []

    for index, data in hybrid_results[:INITIAL_K]:

        candidates.append(
            chunks[index]["text"]
        )

    pairs = [
        [query, text]
        for text in candidates
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for i, score in enumerate(scores):

        index = hybrid_results[i][0]

        reranked.append({
            "index": index,
            "rerank_score": float(score)
        })

    reranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked


def main():

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    query = input("\nEnter your query: ")

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

    print("Running reranker...")

    reranked = rerank(
        query,
        hybrid_results,
        chunks,
        reranker
    )

    print("\n===== FINAL RERANKED RESULTS =====")

    for rank, result in enumerate(
        reranked[:FINAL_K],
        start=1
    ):

        chunk = chunks[result["index"]]

        print(f"\nRank: {rank}")
        print(
            f"Rerank Score: "
            f"{result['rerank_score']:.4f}"
        )
        print(f"Chunk: {chunk['chunk_id']}")
        print(f"Document: {chunk['source_file']}")
        print(f"Page: {chunk['page']}")
        print(f"Text: {chunk['text'][:500]}...")


if __name__ == "__main__":
    main()