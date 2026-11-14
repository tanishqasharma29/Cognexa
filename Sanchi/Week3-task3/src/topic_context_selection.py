import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# -----------------------------
# SETTINGS
# -----------------------------

TOPICS = [
    "RAG",
    "Transformers",
    "Hashing"
]

TOP_K_RETRIEVAL = 3
MAX_CHUNKS = 5
MAX_CONTEXT_WORDS = 1800
RRF_K = 60


# -----------------------------
# LOAD CHUNKS
# -----------------------------

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]


# -----------------------------
# LOAD MODELS
# -----------------------------

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# -----------------------------
# BM25
# -----------------------------

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)


# -----------------------------
# DENSE EMBEDDINGS
# -----------------------------

print("Creating embeddings...")

embeddings = embedder.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=False
)


# -----------------------------
# TOPIC RETRIEVAL
# -----------------------------

def retrieve_for_topic(topic):

    print(f"\nRetrieving for topic: {topic}")

    # Dense retrieval
    query_embedding = embedder.encode(
        [topic],
        normalize_embeddings=True
    )[0]

    dense_scores = np.dot(embeddings, query_embedding)

    dense_indices = np.argsort(
        dense_scores
    )[::-1][:TOP_K_RETRIEVAL]

    # BM25 retrieval
    bm25_scores = bm25.get_scores(
        topic.lower().split()
    )

    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:TOP_K_RETRIEVAL]

    # -----------------------------
    # RRF
    # -----------------------------

    rrf_scores = {}

    for rank, idx in enumerate(dense_indices, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (
            1 / (RRF_K + rank)
        )

    for rank, idx in enumerate(bm25_indices, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (
            1 / (RRF_K + rank)
        )

    candidates = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    candidate_indices = [
        idx for idx, _ in candidates
    ]

    # -----------------------------
    # RERANK
    # -----------------------------

    pairs = [
        [topic, texts[idx]]
        for idx in candidate_indices
    ]

    rerank_scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_indices, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked:

        results.append({
            "index": int(idx),
            "rerank_score": float(score),
            "rrf_score": float(rrf_scores[idx])
        })

    return results


# -----------------------------
# RETRIEVE FOR EACH TOPIC
# -----------------------------

topic_results = {}

for topic in TOPICS:
    topic_results[topic] = retrieve_for_topic(topic)


# -----------------------------
# SHOW RESULTS
# -----------------------------

print("\n" + "=" * 70)
print("TOPIC RETRIEVAL RESULTS")
print("=" * 70)

for topic, results in topic_results.items():

    print(f"\n### {topic}")

    for r in results[:3]:

        chunk = chunks[r["index"]]

        print(
            f"\nChunk: {chunk['chunk_id']}"
        )

        print(
            f"Source: {chunk['source_file']}"
        )

        print(
            f"Page: {chunk['page']}"
        )

        print(
            f"Rerank score: {r['rerank_score']:.4f}"
        )


# -----------------------------
# SELECT CONTEXT
# -----------------------------

selected = []
selected_ids = set()
total_words = 0


# First guarantee at least ONE chunk
# from every requested topic.

for topic in TOPICS:

    if not topic_results[topic]:
        continue

    best = topic_results[topic][0]

    idx = best["index"]
    chunk = chunks[idx]

    if chunk["chunk_id"] in selected_ids:
        continue

    words = chunk["word_count"]

    if total_words + words > MAX_CONTEXT_WORDS:
        continue

    selected.append({
        "topic": topic,
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source_file": chunk["source_file"],
        "page": chunk["page"],
        "heading": chunk.get("heading", ""),
        "text": chunk["text"],
        "word_count": words,
        "rerank_score": best["rerank_score"]
    })

    selected_ids.add(chunk["chunk_id"])
    total_words += words


# -----------------------------
# FILL REMAINING SLOTS
# -----------------------------

remaining = []

for topic, results in topic_results.items():

    for r in results:

        idx = r["index"]
        chunk = chunks[idx]

        if chunk["chunk_id"] in selected_ids:
            continue

        remaining.append({
            "topic": topic,
            "chunk": chunk,
            "rerank_score": r["rerank_score"]
        })


remaining.sort(
    key=lambda x: x["rerank_score"],
    reverse=True
)


for item in remaining:

    if len(selected) >= MAX_CHUNKS:
        break

    chunk = item["chunk"]

    if chunk["chunk_id"] in selected_ids:
        continue

    words = chunk["word_count"]

    if total_words + words > MAX_CONTEXT_WORDS:
        continue

    selected.append({
        "topic": item["topic"],
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "source_file": chunk["source_file"],
        "page": chunk["page"],
        "heading": chunk.get("heading", ""),
        "text": chunk["text"],
        "word_count": words,
        "rerank_score": item["rerank_score"]
    })

    selected_ids.add(chunk["chunk_id"])
    total_words += words


# -----------------------------
# SAVE
# -----------------------------

output = {
    "topics": TOPICS,
    "selected_chunks": selected,
    "total_chunks": len(selected),
    "total_words": total_words
}

with open(
    "results/selected_context.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


# -----------------------------
# FINAL SUMMARY
# -----------------------------

print("\n" + "=" * 70)
print("FINAL SELECTED CONTEXT")
print("=" * 70)

for item in selected:

    print(
        f"{item['topic']:12} | "
        f"{item['chunk_id']:15} | "
        f"{item['source_file']:10} | "
        f"page {item['page']}"
    )

print("\nTotal chunks:", len(selected))
print("Total words:", total_words)

print(
    "\nSaved to: results/selected_context.json"
)