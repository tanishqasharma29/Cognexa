import json
from pathlib import Path
import re


INPUT_FILE = Path("data/processed/extracted_pages.json")
OUTPUT_FILE = Path("data/processed/chunks.json")

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
    """
    Simple heuristic heading detection.
    This is intentionally lightweight for the prototype.
    """

    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        if 2 <= len(line.split()) <= 10:
            if len(line) < 100:
                return line

    return None


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

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

            chunk_id = (
                f'{page["document_id"]}'
                f'_p{page["page"]}'
                f'_c{index + 1}'
            )

            chunk = {
                "chunk_id": chunk_id,
                "document_id": page["document_id"],
                "source_file": page["source_file"],
                "page": page["page"],
                "heading": heading,
                "text": chunk_text,
                "word_count": len(chunk_text.split())
            }

            all_chunks.append(chunk)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Created {len(all_chunks)} chunks")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()