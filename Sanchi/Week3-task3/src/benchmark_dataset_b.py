
import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_NAME = "B"

CHUNKS_FILE = BASE_DIR / "data/processed/dataset_B_actual/chunks.json"
QUERIES_FILE = BASE_DIR / "results/benchmark_queries.json"
INDEX_FILE = BASE_DIR / "indexes/dense_index_actual_dataset_b.npz"

OUTPUT_FILE = BASE_DIR / "results/benchmark_results_actual_B.json"

TOP_K = 5
CANDIDATE_K = 10
RRF_K = 60

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_data():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not chunks:
        raise ValueError("No chunks found in Dataset B.")

    if not queries:
        raise ValueError("No benchmark queries found.")

    chunk_ids = [c["chunk_id"] for c in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found.")

    return chunks, queries


def load_dense_index(chunks):
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Dense index not found: {INDEX_FILE}"
        )

    with np.load(INDEX_FILE, allow_pickle=True) as data:
        if "embeddings" not in data.files or "chunk_ids" not in data.files:
            raise ValueError(
                f"Unexpected dense index format. Available keys: {data.files}"
            )

        embeddings = np.asarray(data["embeddings"])
        indexed_chunk_ids = data["chunk_ids"].astype(str).tolist()

    current_chunk_ids = [str(c["chunk_id"]) for c in chunks]

    if indexed_chunk_ids != current_chunk_ids:
        raise ValueError(
            "Dense index chunk IDs do not match Dataset B chunks.json. "
            "Regenerate the Dataset B dense index."
        )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Embedding count does not match the number of Dataset B chunks."
        )

    return embeddings


def validate_queries(queries, chunks):
    available_ids = {str(c["chunk_id"]) for c in chunks}

    for query in queries:
        expected = query.get("expected_chunks", [])

        missing = [
            cid for cid in expected
            if str(cid) not in available_ids
        ]

        if missing:
            raise ValueError(
                f"Query {query.get('id')} contains expected chunk IDs "
                f"not found in actual Dataset B: {missing}"
            )


def get_metadata(chunk):
    return {
        "document_id": chunk.get("document_id"),
        "source_file": chunk.get("source_file"),
        "dataset": chunk.get("dataset"),
        "extraction_method": chunk.get("extraction_method"),
        "page": chunk.get("page"),
        "chunk_id": chunk["chunk_id"]
    }


def build_bm25(chunks):
    tokenized = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def lexical_search(query, bm25, chunks):
    scores = bm25.get_scores(tokenize(query))
    ranked = np.argsort(scores)[::-1][:CANDIDATE_K]

    return [
        {
            **get_metadata(chunks[i]),
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
            **get_metadata(chunks[i]),
            "score": float(scores[i]),
            "rank": rank + 1
        }
        for rank, i in enumerate(ranked)
    ]


def hybrid_search(lexical, dense, chunks):
    scores = {}
    chunk_map = {c["chunk_id"]: c for c in chunks}

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
            **get_metadata(chunk_map[cid]),
            "score": float(score),
            "rank": rank + 1
        }
        for rank, (cid, score) in enumerate(ranked)
    ]


def rerank(query, hybrid_results, chunks, reranker):
    chunk_map = {c["chunk_id"]: c for c in chunks}

    pairs = [
        [query, chunk_map[item["chunk_id"]]["text"]]
        for item in hybrid_results
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(hybrid_results, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )[:CANDIDATE_K]

    return [
        {
            **{
                key: value
                for key, value in item.items()
                if key not in ("score", "rank")
            },
            "score": float(score),
            "rank": rank + 1
        }
        for rank, (item, score) in enumerate(ranked)
    ]


def hit_at_k(results, expected_chunks, k):
    if not expected_chunks:
        return None

    returned = {
        str(item["chunk_id"])
        for item in results[:k]
    }

    expected = {str(cid) for cid in expected_chunks}

    return int(bool(returned.intersection(expected)))


def recall_at_k(results, expected_chunks, k):
    if not expected_chunks:
        return None

    returned = {
        str(item["chunk_id"])
        for item in results[:k]
    }

    expected = {str(cid) for cid in expected_chunks}

    relevant_found = returned.intersection(expected)

    return len(relevant_found) / len(expected)


def precision_at_k(results, expected_chunks, k):
    if not expected_chunks:
        return None

    returned = {
        str(item["chunk_id"])
        for item in results[:k]
    }

    expected = {str(cid) for cid in expected_chunks}

    relevant_found = returned.intersection(expected)

    return len(relevant_found) / k


def reciprocal_rank(results, expected_chunks):
    if not expected_chunks:
        return None

    expected = {str(cid) for cid in expected_chunks}

    for item in results[:TOP_K]:
        if str(item["chunk_id"]) in expected:
            return 1 / item["rank"]

    return 0.0


def unsupported_behavior(results, expected_chunks):
    if expected_chunks:
        return None

    return len(results[:TOP_K]) == 0


def main():
    print("\nLoading actual Dataset B benchmark data...")

    chunks, queries = load_data()

    print(f"Dataset: {DATASET_NAME}")
    print(f"Chunks: {len(chunks)}")
    print(f"Queries: {len(queries)}")

    print("\nValidating expected chunk IDs...")
    validate_queries(queries, chunks)
    print("All expected chunk IDs are valid.")

    print("\nLoading saved dense index...")
    embeddings = load_dense_index(chunks)
    print(f"Dense embeddings shape: {embeddings.shape}")

    print("\nBuilding BM25 index...")
    bm25 = build_bm25(chunks)

    print("\nLoading embedding model...")
    dense_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nLoading reranker...")
    reranker = CrossEncoder(RERANKER_MODEL)

    methods = [
        "lexical",
        "dense",
        "hybrid",
        "hybrid_reranking"
    ]

    results = []

    for q in queries:
        print(f"\nRunning {q['id']}: {q['query']}")

        lexical_start = time.perf_counter()
        lexical = lexical_search(q["query"], bm25, chunks)
        lexical_time = time.perf_counter() - lexical_start

        dense_start = time.perf_counter()
        dense = dense_search(
            q["query"],
            dense_model,
            embeddings,
            chunks
        )
        dense_time = time.perf_counter() - dense_start

        hybrid_start = time.perf_counter()
        hybrid = hybrid_search(lexical, dense, chunks)
        hybrid_time = time.perf_counter() - hybrid_start

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
            "expected_chunks": q.get("expected_chunks", []),
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
            expected = q.get("expected_chunks", [])

            query_result["methods"][method] = {
                "results": method_result,
                "top_5": method_result[:TOP_K],
                "hit_at_5": hit_at_k(
                    method_result, expected, 5
                ),
                "recall_at_1": recall_at_k(
                    method_result, expected, 1
                ),
                "recall_at_3": recall_at_k(
                    method_result, expected, 3
                ),
                "recall_at_5": recall_at_k(
                    method_result, expected, 5
                ),
                "precision_at_1": precision_at_k(
                    method_result, expected, 1
                ),
                "precision_at_3": precision_at_k(
                    method_result, expected, 3
                ),
                "precision_at_5": precision_at_k(
                    method_result, expected, 5
                ),
                "recall_at_10": recall_at_k(
                    method_result, expected, 10
                ),
                "precision_at_10": precision_at_k(
                    method_result, expected, 10
                ),
                "mrr": reciprocal_rank(
                    method_result, expected
                ),
                "unsupported_clean": unsupported_behavior(
                    method_result, expected
                ),
                "time_seconds": times[method]
            }

        results.append(query_result)

        print(
            f"Completed {q['id']} | "
            f"Lexical: {len(lexical)} | "
            f"Dense: {len(dense)} | "
            f"Hybrid: {len(hybrid)} | "
            f"Reranked: {len(reranked)}"
        )

    output = {
        "benchmark": {
            "dataset": DATASET_NAME,
            "dataset_type": "actual",
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "reranker": RERANKER_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "dense_index": str(INDEX_FILE),
            "num_chunks": len(chunks),
            "num_queries": len(queries)
        },
        "queries": results
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nActual Dataset B benchmark complete.")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total queries: {len(queries)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
