
import json
from pathlib import Path

CHUNKS_A = Path("data/processed/chunks.json")
OCR_FILE = Path("data/processed/d4_ocr_pages.json")
OUTPUT_FILE = Path("data/processed/chunks_dataset_b.json")


def main():
    with open(CHUNKS_A, "r", encoding="utf-8") as f:
        chunks_a = json.load(f)

    with open(OCR_FILE, "r", encoding="utf-8") as f:
        ocr_pages = json.load(f)

    chunks_b = list(chunks_a)

    for page in ocr_pages:
        text = page["text"].strip()

        if not text:
            print(f"Skipping empty page {page['page']}")
            continue

        chunks_b.append({
            "chunk_id": f"d4_p{page['page']}_c1",
            "document_id": "d4",
            "source_file": "d4.pdf",
            "page": page["page"],
            "text": text,
            "character_count": len(text),
            "word_count": len(text.split()),
            "extraction_method": "tesseract_ocr"
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_b, f, indent=2, ensure_ascii=False)

    print("\nDataset B created successfully.")
    print(f"Dataset A chunks: {len(chunks_a)}")
    print(f"OCR pages added: {len(chunks_b) - len(chunks_a)}")
    print(f"Dataset B total chunks: {len(chunks_b)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
