import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


CHUNKS_FILE = Path("data/processed/chunks.json")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bm25(chunks):
    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized_chunks)


def search(query, chunks, bm25, top_k=5):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):

        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "chunk_id": chunks[index]["chunk_id"],
            "document_id": chunks[index]["document_id"],
            "source_file": chunks[index]["source_file"],
            "page": chunks[index]["page"],
            "text": chunks[index]["text"]
        })

    return results


def main():

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    bm25 = build_bm25(chunks)

    query = input("\nEnter your query: ")

    results = search(
        query,
        chunks,
        bm25,
        top_k=5
    )

    print("\n===== BM25 LEXICAL RETRIEVAL =====")

    for result in results:

        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Document: {result['source_file']}")
        print(f"Page: {result['page']}")
        print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    main()