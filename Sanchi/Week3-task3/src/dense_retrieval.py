
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]

DATASETS = {
    "A": BASE_DIR / "data/processed/dataset_A_actual",
    "B": BASE_DIR / "data/processed/dataset_B_actual"
}

INDEX_DIR = BASE_DIR / "indexes"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5

def load_chunks(dataset):
    chunks_file = DATASETS[dataset] / "chunks.json"

    with open(chunks_file, "r", encoding="utf-8") as file:
        return json.load(file)

def build_index(chunks, model):
    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.asarray(embeddings)

def save_index(dataset, chunks, embeddings):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_file = INDEX_DIR / f"dense_index_actual_dataset_{dataset.lower()}.npz"

    np.savez_compressed(
        index_file,
        embeddings=embeddings,
        chunk_ids=np.array([chunk["chunk_id"] for chunk in chunks])
    )

    return index_file

def search(query, chunks, embeddings, model, top_k=TOP_K):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]

        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "source_file": chunk["source_file"],
            "page": chunk["page"],
            "text": chunk["text"]
        })

    return results

def main():
    print("Available datasets: A and B")
    dataset = input("Select dataset (A/B): ").strip().upper()

    if dataset not in DATASETS:
        raise ValueError("Invalid dataset. Enter A or B.")

    chunks = load_chunks(dataset)

    print(f"\nDataset {dataset}: Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = build_index(chunks, model)

    index_file = save_index(dataset, chunks, embeddings)

    print("\nDense index created.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {index_file}")

    query = input("\nEnter your query: ")

    results = search(query, chunks, embeddings, model)

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
