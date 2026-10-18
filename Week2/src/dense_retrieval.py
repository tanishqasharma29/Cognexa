import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_FILE = Path("indexes/dense_index.npz")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.array(embeddings)


def save_index(embeddings):
    np.savez_compressed(
        INDEX_FILE,
        embeddings=embeddings
    )


def search(query, chunks, embeddings, model, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

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

    model = SentenceTransformer(MODEL_NAME)

    embeddings = build_index(chunks, model)

    save_index(embeddings)

    print()
    print("Dense index created.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {INDEX_FILE}")

    query = input("\nEnter your query: ")

    results = search(
        query,
        chunks,
        embeddings,
        model,
        top_k=5
    )

    print("\n===== DENSE RETRIEVAL RESULTS =====")

    for result in results:

        print(f"\nRank: {result['rank']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Document: {result['source_file']}")
        print(f"Page: {result['page']}")
        print(f"Text: {result['text'][:500]}...")


if __name__ == "__main__":
    main()