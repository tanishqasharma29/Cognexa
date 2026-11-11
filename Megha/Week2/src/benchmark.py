import json
import time
import argparse
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 10
RRF_CONSTANT = 60
K_VALUES = [1, 3, 5, 10]


def tokenize(text):
    return text.lower().split()


def precision_at_k(results, relevant, k):
    hits = sum(
        result["chunk_id"] in relevant
        for result in results[:k]
    )
    return hits / k


def recall_at_k(results, relevant, k):
    if not relevant:
        return 0.0

    hits = sum(
        result["chunk_id"] in relevant
        for result in results[:k]
    )

    return hits / len(relevant)


def mrr(results, relevant):
    for rank, result in enumerate(results, start=1):
        if result["chunk_id"] in relevant:
            return 1 / rank

    return 0.0


def make_results(indices, scores, chunks):
    results = []

    for idx in indices:
        idx = int(idx)
        chunk = chunks[idx]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(scores[idx])
        })

    return results


def evaluate_method(results, relevant):
    metrics = {}

    for k in K_VALUES:
        metrics[f"precision@{k}"] = precision_at_k(
            results, relevant, k
        )

        metrics[f"recall@{k}"] = recall_at_k(
            results, relevant, k
        )

    metrics["mrr"] = mrr(results, relevant)

    return metrics


def run_benchmark(chunks_path, output_path, dataset):
    chunks_path = Path(chunks_path)
    output_path = Path(output_path)
    query_path = Path("results/benchmark_queries.json")

    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)

    if not query_path.exists():
        raise FileNotFoundError(query_path)

    with chunks_path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    with query_path.open("r", encoding="utf-8") as f:
        queries = json.load(f)

    if not chunks:
        raise ValueError("No chunks found.")

    expected_field = f"expected_chunks_{dataset}"

    available_ids = {
        chunk["chunk_id"] for chunk in chunks
    }

    for query in queries:
        missing = set(query.get(expected_field, [])) - available_ids

        if missing:
            raise ValueError(
                f"Invalid ground-truth IDs for {query['query_id']}: "
                f"{sorted(missing)}"
            )

    print(f"\nLoading Dataset {dataset.upper()}")
    print(f"Total chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("Building BM25 index...")
    bm25 = BM25Okapi([
        tokenize(text) for text in texts
    ])

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading reranker...")
    cross_encoder = CrossEncoder(RERANKER_MODEL)

    print("Creating document embeddings...")
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings)

    all_query_results = []

    for query in queries:
        query_id = query["query_id"]
        query_text = query["query"]

        relevant = set(query.get(expected_field, []))

        if not relevant:
            all_query_results.append({
                "query_id": query_id,
                "query": query_text,
                "answerable": False,
                "methods": {},
                "llm_context_chunks": []
            })
            continue

        print(f"\nQuery: {query_id} - {query_text}")

        # BM25
        start = time.perf_counter()

        bm25_scores = np.asarray(
            bm25.get_scores(tokenize(query_text))
        )

        bm25_indices = np.argsort(
            bm25_scores
        )[::-1][:TOP_K]

        bm25_results = make_results(
            bm25_indices,
            bm25_scores,
            chunks
        )

        bm25_time = time.perf_counter() - start

        # Dense retrieval
        start = time.perf_counter()

        query_embedding = embedding_model.encode(
            [query_text],
            normalize_embeddings=True
        )[0]

        dense_scores = embeddings @ query_embedding

        dense_indices = np.argsort(
            dense_scores
        )[::-1][:TOP_K]

        dense_results = make_results(
            dense_indices,
            dense_scores,
            chunks
        )

        dense_time = time.perf_counter() - start

        # Hybrid retrieval: Reciprocal Rank Fusion
        start = time.perf_counter()

        rrf_scores = {}

        for rank, idx in enumerate(bm25_indices, start=1):
            idx = int(idx)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (
                1 / (RRF_CONSTANT + rank)
            )

        for rank, idx in enumerate(dense_indices, start=1):
            idx = int(idx)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (
                1 / (RRF_CONSTANT + rank)
            )

        hybrid_indices = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )[:TOP_K]

        hybrid_results = make_results(
            hybrid_indices,
            rrf_scores,
            chunks
        )

        fusion_time = time.perf_counter() - start
        hybrid_time = bm25_time + dense_time + fusion_time

        # Hybrid + Cross-Encoder reranking
        start = time.perf_counter()

        pairs = [
            (query_text, chunks[idx]["text"])
            for idx in hybrid_indices
        ]

        rerank_scores = cross_encoder.predict(pairs)

        reranked_order = np.argsort(
            rerank_scores
        )[::-1]

        reranked_indices = [
            hybrid_indices[int(i)]
            for i in reranked_order
        ]

        rerank_score_map = {
            hybrid_indices[i]: float(rerank_scores[i])
            for i in range(len(hybrid_indices))
        }

        reranked_results = make_results(
            reranked_indices,
            rerank_score_map,
            chunks
        )

        reranker_time = time.perf_counter() - start
        total_rerank_time = hybrid_time + reranker_time

        method_results = {
            "bm25": (bm25_results, bm25_time),
            "dense": (dense_results, dense_time),
            "hybrid": (hybrid_results, hybrid_time),
            "hybrid_reranking": (
                reranked_results,
                total_rerank_time
            )
        }

        methods_output = {}

        for method, (results, elapsed) in method_results.items():
            metrics = evaluate_method(results, relevant)
            metrics["retrieval_time_seconds"] = float(elapsed)

            methods_output[method] = {
                "metrics": metrics,
                "top_results": results[:TOP_K]
            }

        all_query_results.append({
            "query_id": query_id,
            "query": query_text,
            "answerable": True,
            "expected_chunks": sorted(relevant),
            "methods": methods_output,

            # The best five reranked chunks are ready for the LLM.
            "llm_context_chunks": reranked_results[:5]
        })

    answerable = [
        q for q in all_query_results
        if q["answerable"]
    ]

    method_names = [
        "bm25",
        "dense",
        "hybrid",
        "hybrid_reranking"
    ]

    metric_names = (
        [f"precision@{k}" for k in K_VALUES]
        + [f"recall@{k}" for k in K_VALUES]
        + ["mrr", "retrieval_time_seconds"]
    )

    aggregate = {}

    for method in method_names:
        aggregate[method] = {}

        for metric in metric_names:
            values = [
                q["methods"][method]["metrics"][metric]
                for q in answerable
            ]

            aggregate[method][metric] = (
                float(np.mean(values))
                if values else None
            )

    output = {
        "dataset": dataset.upper(),
        "num_chunks": len(chunks),
        "num_queries_total": len(queries),
        "num_queries_answerable": len(answerable),
        "query_coverage": (
            len(answerable) / len(queries)
            if queries else 0
        ),
        "metrics": aggregate,
        "queries": all_query_results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n===== BENCHMARK SUMMARY =====")

    for method, metrics in aggregate.items():
        print(f"\n{method}")

        for metric, value in metrics.items():
            if value is not None:
                print(f"{metric}: {value:.4f}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks",
        default="data/processed/chunks.json"
    )

    parser.add_argument(
        "--output",
        default="results/benchmark_results.json"
    )

    parser.add_argument(
        "--dataset",
        choices=["a", "b"],
        default="a"
    )

    args = parser.parse_args()

    run_benchmark(
        args.chunks,
        args.output,
        args.dataset
    )