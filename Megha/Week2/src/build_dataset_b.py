
import json
import re
from pathlib import Path

INPUT_FILE = Path("data/processed/extracted_pages_b.json")
OUTPUT_FILE = Path("data/processed/chunks_dataset_b.json")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def split_into_chunks(text):
    words = re.sub(r"\s+", " ", text).strip().split()

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run extract.py first.")

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []

    for page in pages:
        page_chunks = split_into_chunks(page["text"])

        for index, text in enumerate(page_chunks, start=1):
            chunk_id = f"{page['document_id']}_p{page['page']}_c{index}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "document_id": page["document_id"],
                "source_file": page["source_file"],
                "source_path": page["source_path"],
                "page": page["page"],
                "text": text,
                "extraction_method": page["extraction_method"]
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    selectable = sum(
        c["extraction_method"] == "text"
        for c in all_chunks
    )

    ocr = sum(
        c["extraction_method"] == "ocr"
        for c in all_chunks
    )

    print("Dataset B chunking complete.")
    print(f"Selectable chunks: {selectable}")
    print(f"OCR chunks: {ocr}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
