import fitz
import json
from pathlib import Path


INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        pages.append({
            "document_id": pdf_path.stem,
            "source_file": pdf_path.name,
            "page": page_number,
            "text": text.strip(),
            "character_count": len(text),
            "word_count": len(text.split()),
        })

    return pages


def main():
    all_pages = []

    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw/")
        return

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        pages = extract_pdf(pdf_path)
        all_pages.extend(pages)

        print(f"  Pages extracted: {len(pages)}")

    output_file = OUTPUT_DIR / "extracted_pages.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)

    print()
    print(f"Saved: {output_file}")
    print(f"Total pages: {len(all_pages)}")


if __name__ == "__main__":
    main()