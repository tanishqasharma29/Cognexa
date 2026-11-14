
import csv
import json
import re
import html
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

EXTRACTED_FILE = BASE_DIR / "data/processed/extracted_pages.json"
D4_OCR_FILE = BASE_DIR / "data/processed/d4_ocr_pages.json"
ADDITIONAL_OCR_FILE = BASE_DIR / "deliverables/page_ocr_results.csv"

OUTPUT_A = BASE_DIR / "data/processed/dataset_A"
OUTPUT_B = BASE_DIR / "data/processed/dataset_B"

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

            document_id = page["document_id"]
            page_number = int(page["page"])

            chunks.append({
                "chunk_id": f"{document_id}_p{page_number:04d}_c{chunk_number:04d}",
                "document_id": document_id,
                "source_file": page["source_file"],
                "page": page_number,
                "text": chunk_text,
                "word_count": len(chunk_text.split()),
                "extraction_method": page.get("extraction_method", "pdf_text")
            })

            if end == len(words):
                break

            start = end - CHUNK_OVERLAP
            chunk_number += 1

    return chunks

def load_original_pages():
    if not EXTRACTED_FILE.exists():
        raise FileNotFoundError(f"Missing file: {EXTRACTED_FILE}")

    if not D4_OCR_FILE.exists():
        raise FileNotFoundError(f"Missing file: {D4_OCR_FILE}")

    pages = load_json(EXTRACTED_FILE)

    original_pages = [
        {
            **page,
            "extraction_method": page.get("extraction_method", "pdf_text")
        }
        for page in pages
        if page["document_id"] in {"d1", "d2", "d3"}
    ]

    d4_pages = load_json(D4_OCR_FILE)

    for page in d4_pages:
        original_pages.append({
            **page,
            "document_id": "d4",
            "source_file": "d4.pdf",
            "page": int(page["page"]),
            "extraction_method": "tesseract_ocr"
        })

    document_ids = {page["document_id"] for page in original_pages}

    if document_ids != {"d1", "d2", "d3", "d4"}:
        raise ValueError(f"Unexpected original document IDs: {document_ids}")

    return original_pages

def load_additional_ocr_pages():
    if not ADDITIONAL_OCR_FILE.exists():
        raise FileNotFoundError(
            "page_ocr_results.csv is missing. "
            "Extract the OCR deliverables folder first."
        )

    pages = []

    with open(
        ADDITIONAL_OCR_FILE,
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
            raise ValueError(
                f"CSV must contain columns: {required_columns}"
            )

        for row in reader:
            text = (row.get("preprocessed_text") or "").strip()

            if not text:
                text = (row.get("original_text") or "").strip()

            pages.append({
                "document_id": row["document_id"].strip(),
                "source_file": row["document_name"].strip(),
                "page": int(row["page_number"]),
                "text": text,
                "extraction_method": "ocr_preprocessed"
            })

    document_ids = {page["document_id"] for page in pages}

    if len(document_ids) != 9:
        raise ValueError(
            f"Expected 9 additional OCR documents, found {len(document_ids)}."
        )

    return pages

def validate_pages(pages, dataset_name):
    keys = [
        (page["document_id"], int(page["page"]))
        for page in pages
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate document/page entries in {dataset_name}")

    empty_pages = sum(
        1 for page in pages if not clean_text(page.get("text", ""))
    )

    print(f"\n{dataset_name} validation")
    print("Documents:", len({p["document_id"] for p in pages}))
    print("Pages:", len(pages))
    print("Empty pages:", empty_pages)

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def main():
    original_pages = load_original_pages()
    additional_pages = load_additional_ocr_pages()

    pages_a = original_pages
    pages_b = pages_a + additional_pages

    validate_pages(pages_a, "Dataset A")
    validate_pages(pages_b, "Dataset B")

    chunks_a = make_chunks(pages_a)
    chunks_b = make_chunks(pages_b)

    OUTPUT_A.mkdir(parents=True, exist_ok=True)
    OUTPUT_B.mkdir(parents=True, exist_ok=True)

    save_json(OUTPUT_A / "pages.json", pages_a)
    save_json(OUTPUT_A / "chunks.json", chunks_a)

    save_json(OUTPUT_B / "pages.json", pages_b)
    save_json(OUTPUT_B / "chunks.json", chunks_b)

    print("\nDATASET BUILD COMPLETE")
    print("----------------------")
    print("Original documents:", len({p["document_id"] for p in original_pages}))
    print("Additional OCR documents:", len({p["document_id"] for p in additional_pages}))
    print("Dataset A pages:", len(pages_a))
    print("Dataset A chunks:", len(chunks_a))
    print("Dataset B pages:", len(pages_b))
    print("Dataset B chunks:", len(chunks_b))
    print("\nDataset A:", OUTPUT_A)
    print("Dataset B:", OUTPUT_B)

if __name__ == "__main__":
    main()
