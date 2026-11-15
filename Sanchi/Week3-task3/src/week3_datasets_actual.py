
import csv
import json
import re
import html
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

ACTUAL_PAGES_FILE = BASE_DIR / "data/processed/extracted_pages_actual.json"
OCR_FILE = BASE_DIR / "deliverables/page_ocr_results.csv"

OUTPUT_A = BASE_DIR / "data/processed/dataset_A_actual"
OUTPUT_B = BASE_DIR / "data/processed/dataset_B_actual"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_heading(text):
    for line in text.split("\n"):
        line = line.strip()

        if 2 <= len(line.split()) <= 10 and len(line) < 100:
            return line

    return None


def make_chunks(pages):
    chunks = []

    for page in pages:
        text = clean_text(page.get("text", ""))

        if not text:
            continue

        words = text.split()
        start = 0
        chunk_number = 1

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])

            chunks.append({
                "chunk_id": (
                    f'{page["document_id"]}'
                    f'_p{int(page["page"]):04d}'
                    f'_c{chunk_number:04d}'
                ),
                "document_id": page["document_id"],
                "source_file": page["source_file"],
                "source_path": page.get("source_path", ""),
                "dataset": page.get("dataset", "ocr"),
                "page": int(page["page"]),
                "heading": detect_heading(page.get("text", "")),
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "extraction_method": page.get(
                    "extraction_method", "pdf_text"
                )
            })

            if end == len(words):
                break

            start = end - CHUNK_OVERLAP
            chunk_number += 1

    return chunks


def load_actual_pages():
    if not ACTUAL_PAGES_FILE.exists():
        raise FileNotFoundError(ACTUAL_PAGES_FILE)

    pages = load_json(ACTUAL_PAGES_FILE)

    for page in pages:
        page["extraction_method"] = "pdf_text"

    return pages


def load_additional_ocr_pages():
    if not OCR_FILE.exists():
        raise FileNotFoundError(OCR_FILE)

    pages = []

    with open(
        OCR_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "document_id",
            "document_name",
            "page_number"
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Required OCR CSV columns are missing.")

        for row in reader:
            text = (row.get("preprocessed_text") or "").strip()

            if not text:
                text = (row.get("original_text") or "").strip()

            pages.append({
                "document_id": row["document_id"].strip(),
                "source_file": row["document_name"].strip(),
                "source_path": "deliverables/page_ocr_results.csv",
                "dataset": "ocr",
                "page": int(row["page_number"]),
                "text": text,
                "extraction_method": "ocr_preprocessed"
            })

    document_ids = {
        page["document_id"] for page in pages
    }

    if len(document_ids) != 9:
        raise ValueError(
            f"Expected 9 OCR documents, found {len(document_ids)}."
        )

    return pages


def validate_pages(pages, name):
    keys = [
        (page["document_id"], int(page["page"]))
        for page in pages
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate document/page entries in {name}")

    empty_pages = sum(
        1 for page in pages
        if not clean_text(page.get("text", ""))
    )

    print(f"\n{name} validation")
    print("Unique documents:", len({
        page["document_id"] for page in pages
    }))
    print("Pages:", len(pages))
    print("Empty-text pages:", empty_pages)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    actual_pages = load_actual_pages()
    ocr_pages = load_additional_ocr_pages()

    actual_ids = {
        page["document_id"] for page in actual_pages
    }

    ocr_ids = {
        page["document_id"] for page in ocr_pages
    }

    if actual_ids.intersection(ocr_ids):
        raise ValueError("OCR document IDs overlap with actual PDF IDs.")

    pages_a = actual_pages
    pages_b = actual_pages + ocr_pages

    validate_pages(pages_a, "Dataset A")
    validate_pages(pages_b, "Dataset B")

    chunks_a = make_chunks(pages_a)
    chunks_b = make_chunks(pages_b)

    save_json(OUTPUT_A / "pages.json", pages_a)
    save_json(OUTPUT_A / "chunks.json", chunks_a)

    save_json(OUTPUT_B / "pages.json", pages_b)
    save_json(OUTPUT_B / "chunks.json", chunks_b)

    print("\nACTUAL DATASET BUILD COMPLETE")
    print("-----------------------------")
    print("Original PDF documents extracted:", len(actual_ids))
    print("Additional OCR documents:", len(ocr_ids))
    print("Dataset A pages:", len(pages_a))
    print("Dataset A chunks:", len(chunks_a))
    print("Dataset B pages:", len(pages_b))
    print("Dataset B chunks:", len(chunks_b))
    print("\nDataset A:", OUTPUT_A)
    print("Dataset B:", OUTPUT_B)


if __name__ == "__main__":
    main()
