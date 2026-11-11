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

MAX_CONTEXT_WORDS = 1800


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def dense_search(query, chunks, model):

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

    indices = np.argsort(scores)[::-1][:INITIAL_K]

    return [
        {
            "rank": rank,
            "score": float(scores[index]),
            "index": int(index)
        }
        for rank, index in enumerate(indices, start=1)
    ]


def bm25_search(query, chunks):

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    scores = bm25.get_scores(
        tokenize(query)
    )

    indices = np.argsort(scores)[::-1][:INITIAL_K]

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
            fused[index] = 0.0

        fused[index] += (
            1 / (RRF_K + result["rank"])
        )

    for result in bm25_results:

        index = result["index"]

        if index not in fused:
            fused[index] = 0.0

        fused[index] += (
            1 / (RRF_K + result["rank"])
        )

    return sorted(
        fused.items(),
        key=lambda x: x[1],
        reverse=True
    )


def rerank(query, hybrid_results, chunks, reranker):

    candidates = []

    for index, _ in hybrid_results[:INITIAL_K]:

        candidates.append(
            chunks[index]["text"]
        )

    pairs = [
        [query, text]
        for text in candidates
    ]

    scores = reranker.predict(pairs)

    results = []

    for i, score in enumerate(scores):

        index = hybrid_results[i][0]

        results.append({
            "index": index,
            "score": float(score)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


def select_context(reranked_results, chunks):

    selected = []
    total_words = 0
    seen_chunks = set()

    for result in reranked_results:

        index = result["index"]
        chunk = chunks[index]

        chunk_id = chunk["chunk_id"]

        # Prevent duplicate chunks
        if chunk_id in seen_chunks:
            continue

        word_count = len(
            chunk["text"].split()
        )

        # Respect context budget
        if total_words + word_count > MAX_CONTEXT_WORDS:
            continue

        selected.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "heading": chunk["heading"],
            "text": chunk["text"],
            "rerank_score": result["score"]
        })

        seen_chunks.add(chunk_id)
        total_words += word_count

        if len(selected) >= FINAL_K:
            break

    return selected


def main():

    chunks = load_chunks()

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    query = input("\nEnter your query: ")

    dense_results = dense_search(
        query,
        chunks,
        embedding_model
    )

    bm25_results = bm25_search(
        query,
        chunks
    )

    hybrid_results = rrf_fusion(
        dense_results,
        bm25_results
    )

    reranked_results = rerank(
        query,
        hybrid_results,
        chunks,
        reranker
    )

    selected_context = select_context(
        reranked_results,
        chunks
    )

    print("\n===== SELECTED CONTEXT =====")

    for i, item in enumerate(
        selected_context,
        start=1
    ):

        print(f"\n--- Evidence {i} ---")
        print(f"Chunk ID: {item['chunk_id']}")
        print(f"Document: {item['source_file']}")
        print(f"Page: {item['page']}")
        print(f"Heading: {item['heading']}")
        print(f"Rerank score: {item['rerank_score']:.4f}")
        print(f"\n{item['text']}")
    with open(
    "results/selected_context.json",
    "w",
    encoding="utf-8"
    ) as f:
        json.dump(
            selected_context,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nSaved selected context to:")
    print("results/selected_context.json")
    print(
        f"\nSelected {len(selected_context)} "
        f"evidence chunks."
    )


if __name__ == "__main__":
    main()