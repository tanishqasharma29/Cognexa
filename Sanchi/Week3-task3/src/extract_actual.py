
import fitz
import json
from pathlib import Path

INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    relative_path = pdf_path.relative_to(INPUT_DIR)
    dataset = relative_path.parts[0]

    document_id = (
        dataset + "_" +
        relative_path.with_suffix("").as_posix().replace("/", "__")
    )

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        pages.append({
            "document_id": document_id,
            "source_file": pdf_path.name,
            "source_path": str(relative_path),
            "dataset": dataset,
            "page": page_number,
            "text": text,
            "character_count": len(text),
            "word_count": len(text.split())
        })

    doc.close()
    return pages


def main():
    all_pages = []
    errors = []

    pdf_files = sorted(INPUT_DIR.rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    print("Total PDFs found:", len(pdf_files))

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{index}/{len(pdf_files)}] Processing: {pdf_path.name}")

        try:
            pages = extract_pdf(pdf_path)
            all_pages.extend(pages)
            print("  Pages extracted:", len(pages))

        except Exception as error:
            errors.append({
                "file": str(pdf_path),
                "error": str(error)
            })
            print("  Extraction failed:", error)

    output_file = OUTPUT_DIR / "extracted_pages_actual.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_pages, file, indent=2, ensure_ascii=False)

    error_file = OUTPUT_DIR / "extraction_errors_actual.json"

    with open(error_file, "w", encoding="utf-8") as file:
        json.dump(errors, file, indent=2, ensure_ascii=False)

    college_count = sum(
        1 for page in all_pages if page["dataset"] == "college"
    )

    kaggle_count = sum(
        1 for page in all_pages if page["dataset"] == "kaggle"
    )

    print("\nExtraction completed.")
    print("PDF files found:", len(pdf_files))
    print("Total pages extracted:", len(all_pages))
    print("College pages:", college_count)
    print("Kaggle pages:", kaggle_count)
    print("Failed PDFs:", len(errors))
    print("Saved:", output_file)
    print("Error report:", error_file)


if __name__ == "__main__":
    main()
