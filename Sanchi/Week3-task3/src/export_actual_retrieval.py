
import json
import re
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

BASE_DIR = Path(__file__).resolve().parents[1]

QUERIES_FILE = BASE_DIR / "results/benchmark_queries.json"

DATASETS = {
    "A": {
        "chunks": BASE_DIR / "data/processed/dataset_A_actual/chunks.json",
        "index": BASE_DIR / "indexes/dense_index_actual_dataset_a.npz",
        "output": BASE_DIR / "results/actual_retrieval_dataset_a.json"
    },
    "B": {
        "chunks": BASE_DIR / "data/processed/dataset_B_actual/chunks.json",
        "index": BASE_DIR / "indexes/dense_index_actual_dataset_b.npz",
        "output": BASE_DIR / "results/actual_retrieval_dataset_b.json"
    }
}

CANDIDATE_K = 10
RRF_K = 60

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_queries():
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not queries:
        raise ValueError("No benchmark queries found.")

    return queries


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError(f"No chunks found: {path}")

    chunk_ids = [str(c["chunk_id"]) for c in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"Duplicate chunk IDs found: {path}")

    return chunks


def load_dense_index(path, chunks):
    if not path.exists():
        raise FileNotFoundError(f"Dense index not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        embeddings = np.asarray(data["embeddings"])
        indexed_ids = data["chunk_ids"].astype(str).tolist()

    current_ids = [str(c["chunk_id"]) for c in chunks]

    if indexed_ids != current_ids:
        raise ValueError(
            f"Dense index IDs do not match chunks: {path}"
        )

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    return embeddings


def get_metadata(chunk):
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk.get("document_id"),
        "source_file": chunk.get("source_file"),
        "source_path": chunk.get("source_path"),
        "dataset": chunk.get("dataset"),
        "extraction_method": chunk.get("extraction_method"),
        "page": chunk.get("page")
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


def run_dataset(name, config, queries, dense_model, reranker):
    print(f"\nProcessing Dataset {name}...")

    chunks = load_chunks(config["chunks"])
    embeddings = load_dense_index(config["index"], chunks)

    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {embeddings.shape}")

    print("Building BM25...")
    bm25 = build_bm25(chunks)

    output_queries = []

    for query_data in queries:
        query_id = query_data["id"]
        query = query_data["query"]

        print(f"Running {query_id}: {query}")

        start = time.perf_counter()
        lexical = lexical_search(query, bm25, chunks)
        lexical_time = time.perf_counter() - start

        start = time.perf_counter()
        dense = dense_search(
            query, dense_model, embeddings, chunks
        )
        dense_time = time.perf_counter() - start

        start = time.perf_counter()
        hybrid = hybrid_search(lexical, dense, chunks)
        hybrid_time = time.perf_counter() - start

        start = time.perf_counter()
        reranked = rerank(
            query, hybrid, chunks, reranker
        )
        rerank_time = time.perf_counter() - start

        output_queries.append({
            "query_id": query_id,
            "query": query,
            "answerable": query_data.get("answerable"),
            "methods": {
                "lexical": {
                    "results": lexical,
                    "time_seconds": lexical_time
                },
                "dense": {
                    "results": dense,
                    "time_seconds": dense_time
                },
                "hybrid": {
                    "results": hybrid,
                    "time_seconds": hybrid_time
                },
                "hybrid_reranking": {
                    "results": reranked,
                    "time_seconds": rerank_time
                }
            }
        })

        print(f"Completed {query_id}")

    output = {
        "dataset": name,
        "dataset_type": "actual",
        "num_chunks": len(chunks),
        "num_queries": len(queries),
        "candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "queries": output_queries
    }

    config["output"].parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(config["output"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Dataset {name} results saved: {config['output']}")


def main():
    print("Loading benchmark queries...")
    queries = load_queries()

    print(f"Total queries: {len(queries)}")

    print("\nLoading embedding model...")
    dense_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nLoading CrossEncoder...")
    reranker = CrossEncoder(RERANKER_MODEL)

    for name, config in DATASETS.items():
        run_dataset(
            name,
            config,
            queries,
            dense_model,
            reranker
        )

    print("\nBoth actual dataset retrieval exports completed.")


if __name__ == "__main__":
    main()
