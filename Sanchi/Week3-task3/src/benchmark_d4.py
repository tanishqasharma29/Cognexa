
import json
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# Dataset B configuration
CHUNKS_FILE = Path("data/processed/chunks_dataset_b.json")
QUERIES_FILE = Path("results/d4_benchmark_queries.json")
OUTPUT_FILE = Path("results/d4_benchmark_results.json")

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60


def tokenize(text):
    return text.lower().split()


def load_data():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    return chunks, queries


def build_bm25(chunks):
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def build_dense(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return model, np.asarray(embeddings)


def lexical_search(query, bm25, chunks):
    scores = bm25.get_scores(tokenize(query))

    ranked = np.argsort(scores)[::-1][:CANDIDATE_K]

    return [
        {
            "chunk_id": chunks[i]["chunk_id"],
            "score": float(scores[i]),
            "rank": rank + 1
        }
        for rank, i in enumerate(ranked)
    ]


def dense_search(query, model, embeddings, chunks):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    scores = embeddings @ query_embedding

    ranked = np.argsort(scores)[::-1][:CANDIDATE_K]

    return [
        {
            "chunk_id": chunks[i]["chunk_id"],
            "score": float(scores[i]),
            "rank": rank + 1
        }
        for rank, i in enumerate(ranked)
    ]


def hybrid_search(lexical, dense):
    scores = {}

    for item in lexical:
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + (
            1 / (RRF_K + item["rank"])
        )

    for item in dense:
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + (
            1 / (RRF_K + item["rank"])
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:CANDIDATE_K]

    return [
        {
            "chunk_id": cid,
            "score": float(score),
            "rank": rank + 1
        }
        for rank, (cid, score) in enumerate(ranked)
    ]


def rerank(query, hybrid_results, chunks, reranker):
    chunk_map = {
        c["chunk_id"]: c
        for c in chunks
    }

    pairs = [
        [
            query,
            chunk_map[item["chunk_id"]]["text"]
        ]
        for item in hybrid_results
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(hybrid_results, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )[:TOP_K]

    return [
        {
            "chunk_id": item["chunk_id"],
            "score": float(score),
            "rank": rank + 1
        }
        for rank, (item, score) in enumerate(ranked)
    ]


def hit_at_k(results, expected_chunks, k):
    if not expected_chunks:
        return None

    returned = {
        item["chunk_id"]
        for item in results[:k]
    }

    return int(bool(returned.intersection(expected_chunks)))


def recall_at_k(results, expected_chunks, k):
    if not expected_chunks:
        return None

    returned = {
        item["chunk_id"]
        for item in results[:k]
    }

    relevant_found = returned.intersection(expected_chunks)

    return len(relevant_found) / len(expected_chunks)


def precision_at_k(results, expected_chunks, k):
    if not expected_chunks:
        return None

    returned = {
        item["chunk_id"]
        for item in results[:k]
    }

    relevant_found = returned.intersection(expected_chunks)

    return len(relevant_found) / k


def reciprocal_rank(results, expected_chunks):
    if not expected_chunks:
        return None

    for item in results[:TOP_K]:
        if item["chunk_id"] in expected_chunks:
            return 1 / item["rank"]

    return 0.0


def unsupported_behavior(results, expected_chunks):
    if expected_chunks:
        return None

    returned = {
        item["chunk_id"]
        for item in results[:TOP_K]
    }

    return len(returned) == 0


def main():
    print("\nLoading Dataset B benchmark data...")

    chunks, queries = load_data()

    print(f"Dataset B chunks: {len(chunks)}")
    print(f"Queries: {len(queries)}")

    print("\nLoading BM25...")
    bm25 = build_bm25(chunks)

    print("\nLoading dense embedding model...")
    dense_model, embeddings = build_dense(chunks)

    print("\nLoading reranker...")
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    methods = [
        "lexical",
        "dense",
        "hybrid",
        "hybrid_reranking"
    ]

    results = []

    for q in queries:
        print(f"\nRunning {q['id']}: {q['query']}")

        # Lexical retrieval
        lexical_start = time.perf_counter()

        lexical = lexical_search(
            q["query"],
            bm25,
            chunks
        )

        lexical_time = time.perf_counter() - lexical_start

        # Dense retrieval
        dense_start = time.perf_counter()

        dense = dense_search(
            q["query"],
            dense_model,
            embeddings,
            chunks
        )

        dense_time = time.perf_counter() - dense_start

        # Hybrid retrieval
        hybrid_start = time.perf_counter()

        hybrid = hybrid_search(
            lexical,
            dense
        )

        hybrid_time = time.perf_counter() - hybrid_start

        # Reranking
        rerank_start = time.perf_counter()

        reranked = rerank(
            q["query"],
            hybrid,
            chunks,
            reranker
        )

        rerank_time = time.perf_counter() - rerank_start

        method_results = {
            "lexical": lexical,
            "dense": dense,
            "hybrid": hybrid,
            "hybrid_reranking": reranked
        }

        query_result = {
            "query_id": q["id"],
            "query": q["query"],
            "answerable": q["answerable"],
            "expected_chunks": q["expected_chunks"],
            "methods": {}
        }

        times = {
            "lexical": lexical_time,
            "dense": dense_time,
            "hybrid": lexical_time + dense_time + hybrid_time,
            "hybrid_reranking": (
                lexical_time
                + dense_time
                + hybrid_time
                + rerank_time
            )
        }

        for method in methods:
            method_result = method_results[method]

            query_result["methods"][method] = {
                "results": method_result,

                "hit_at_5": hit_at_k(
                    method_result,
                    q["expected_chunks"],
                    5
                ),

                "recall_at_1": recall_at_k(
                    method_result,
                    q["expected_chunks"],
                    1
                ),

                "recall_at_3": recall_at_k(
                    method_result,
                    q["expected_chunks"],
                    3
                ),

                "recall_at_5": recall_at_k(
                    method_result,
                    q["expected_chunks"],
                    5
                ),

                "precision_at_1": precision_at_k(
                    method_result,
                    q["expected_chunks"],
                    1
                ),

                "precision_at_3": precision_at_k(
                    method_result,
                    q["expected_chunks"],
                    3
                ),

                "precision_at_5": precision_at_k(
                    method_result,
                    q["expected_chunks"],
                    5
                ),

                "mrr": reciprocal_rank(
                    method_result,
                    q["expected_chunks"]
                ),

                "unsupported_clean": unsupported_behavior(
                    method_result,
                    q["expected_chunks"]
                ),

                "time_seconds": times[method]
            }

        results.append(query_result)

    output = {
        "benchmark": {
            "dataset": "B",
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "embedding_model": "all-MiniLM-L6-v2",
            "num_chunks": len(chunks),
            "num_queries": len(queries)
        },
        "queries": results
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nDataset B benchmark complete.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
