
import json
from pathlib import Path
import re

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def detect_heading(text):
    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        if 2 <= len(line.split()) <= 10 and len(line) < 100:
            return line

    return None


def main():
    print("\nSelect dataset:")
    print("1. Sample Dataset")
    print("2. Actual Dataset")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        input_file = Path("data/processed/extracted_pages.json")
        output_file = Path("data/processed/chunks.json")
    elif choice == "2":
        input_file = Path("data/processed/extracted_pages_actual.json")
        output_file = Path("data/processed/chunks_actual.json")
    else:
        print("Invalid choice.")
        return

    if not input_file.exists():
        print("Input file not found:", input_file)
        return

    with open(input_file, "r", encoding="utf-8") as file:
        pages = json.load(file)

    all_chunks = []

    for page in pages:
        text = clean_text(page["text"])

        if not text:
            continue

        chunks = split_into_chunks(
            text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        heading = detect_heading(page["text"])

        for index, chunk_text in enumerate(chunks):
            chunk = {
                "chunk_id": (
                    f'{page["document_id"]}'
                    f'_p{page["page"]}'
                    f'_c{index + 1}'
                ),
                "document_id": page["document_id"],
                "source_file": page["source_file"],
                "page": page["page"],
                "heading": heading,
                "text": chunk_text,
                "word_count": len(chunk_text.split())
            }

            if "dataset" in page:
                chunk["dataset"] = page["dataset"]

            if "source_path" in page:
                chunk["source_path"] = page["source_path"]

            all_chunks.append(chunk)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nChunking completed.")
    print("Total pages:", len(pages))
    print("Total chunks:", len(all_chunks))
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()
