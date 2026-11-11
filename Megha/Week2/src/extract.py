
import csv
import hashlib
import json
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

DATASET_A_FILE = PROCESSED_DIR / "extracted_pages_a.json"
DATASET_B_FILE = PROCESSED_DIR / "extracted_pages_b.json"
INVENTORY_FILE = RESULTS_DIR / "dataset_inventory.csv"

CACHE_DIR = PROCESSED_DIR / "extraction_cache"

MIN_TEXT_CHARS = 30
OCR_DPI = 200
OCR_CONFIG = "--oem 1 --psm 6"


def save_json(data, path):
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    temp_path.replace(path)


def get_cache_path(pdf_path):
    relative_path = str(pdf_path.relative_to(RAW_DIR))

    file_info = (
        relative_path
        + str(pdf_path.stat().st_size)
        + str(pdf_path.stat().st_mtime_ns)
    )

    key = hashlib.sha256(file_info.encode()).hexdigest()

    return CACHE_DIR / f"{key}.json"


def extract_pdf(pdf_path, document_id):
    pages = []

    selectable_count = 0
    scanned_count = 0
    ocr_success_count = 0

    relative_path = str(pdf_path.relative_to(ROOT))

    try:
        doc = fitz.open(pdf_path)

        for page_index, page in enumerate(doc):

            text = page.get_text("text").strip()

            if len(text) >= MIN_TEXT_CHARS:
                extraction_method = "text"
                selectable_count += 1

            else:
                scanned_count += 1
                extraction_method = "ocr"

                try:
                    matrix = fitz.Matrix(
                        OCR_DPI / 72,
                        OCR_DPI / 72
                    )

                    pix = page.get_pixmap(
                        matrix=matrix,
                        colorspace=fitz.csGRAY,
                        alpha=False
                    )

                    image = Image.frombytes(
                        "L",
                        [pix.width, pix.height],
                        pix.samples
                    )

                    text = pytesseract.image_to_string(
                        image,
                        config=OCR_CONFIG
                    ).strip()

                    if text:
                        ocr_success_count += 1

                except Exception as e:
                    print(
                        f"\nOCR error: {pdf_path.name}, "
                        f"page {page_index + 1}: {e}"
                    )
                    text = ""

            if text:
                pages.append({
                    "document_id": document_id,
                    "source_file": pdf_path.name,
                    "source_path": relative_path,
                    "page": page_index + 1,
                    "text": text,
                    "extraction_method": extraction_method
                })

        total_pages = len(doc)
        doc.close()

        return {
            "pages": pages,
            "total_pages": total_pages,
            "selectable_pages": selectable_count,
            "scanned_pages": scanned_count,
            "ocr_success_pages": ocr_success_count,
            "status": "success"
        }

    except Exception as e:
        print(f"\nPDF error: {pdf_path.name}: {e}")

        return {
            "pages": [],
            "total_pages": 0,
            "selectable_pages": 0,
            "scanned_pages": 0,
            "ocr_success_pages": 0,
            "status": f"error: {e}"
        }


def main():

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        RAW_DIR.rglob("*.pdf"),
        key=lambda p: str(p).lower()
    )

    print(f"\nTotal PDFs found: {len(pdf_files)}")
    print("Starting extraction...\n")

    dataset_a = []
    dataset_b = []
    inventory = []

    total_selectable = 0
    total_scanned = 0
    total_ocr_success = 0
    total_pages = 0
    failed_pdfs = 0
    cached_pdfs = 0

    for index, pdf_path in enumerate(pdf_files, start=1):

        document_id = f"doc_{index:05d}"
        cache_path = get_cache_path(pdf_path)

        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    result = json.load(f)

                cached_pdfs += 1

            except Exception:
                result = extract_pdf(pdf_path, document_id)
                save_json(result, cache_path)

        else:
            result = extract_pdf(pdf_path, document_id)
            save_json(result, cache_path)

        pages = result["pages"]

        for page in pages:
            dataset_b.append(page)

            if page["extraction_method"] == "text":
                dataset_a.append(page)

        total_pages += result["total_pages"]
        total_selectable += result["selectable_pages"]
        total_scanned += result["scanned_pages"]
        total_ocr_success += result["ocr_success_pages"]

        if result["status"] != "success":
            failed_pdfs += 1

        inventory.append({
            "document_id": document_id,
            "source_file": pdf_path.name,
            "source_path": str(pdf_path.relative_to(ROOT)),
            "total_pages": result["total_pages"],
            "selectable_pages": result["selectable_pages"],
            "scanned_pages": result["scanned_pages"],
            "ocr_success_pages": result["ocr_success_pages"],
            "status": result["status"]
        })

        if index % 10 == 0 or index == len(pdf_files):
            print(
                f"Processed {index}/{len(pdf_files)} PDFs | "
                f"Selectable: {total_selectable} | "
                f"Scanned: {total_scanned} | "
                f"OCR successful: {total_ocr_success} | "
                f"Cached: {cached_pdfs}"
            )

    save_json(dataset_a, DATASET_A_FILE)
    save_json(dataset_b, DATASET_B_FILE)

    with open(INVENTORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "document_id",
                "source_file",
                "source_path",
                "total_pages",
                "selectable_pages",
                "scanned_pages",
                "ocr_success_pages",
                "status"
            ]
        )

        writer.writeheader()
        writer.writerows(inventory)

    print("\n========== EXTRACTION SUMMARY ==========")
    print(f"PDFs processed: {len(pdf_files)}")
    print(f"Failed PDFs: {failed_pdfs}")
    print(f"Cached PDFs reused: {cached_pdfs}")
    print(f"Total pages: {total_pages}")
    print(f"Selectable pages: {total_selectable}")
    print(f"Scanned pages: {total_scanned}")
    print(f"Pages successfully OCR-extracted: {total_ocr_success}")
    print(f"Dataset A pages: {len(dataset_a)}")
    print(f"Dataset B pages: {len(dataset_b)}")

    print("\nFiles saved:")
    print(DATASET_A_FILE)
    print(DATASET_B_FILE)
    print(INVENTORY_FILE)


if __name__ == "__main__":
    main()
