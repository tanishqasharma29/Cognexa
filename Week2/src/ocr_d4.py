
import json
import fitz
import cv2
import numpy as np
import pytesseract

from pathlib import Path


PDF_FILE = Path("data/raw/d4.pdf")
OUTPUT_FILE = Path("data/processed/d4_ocr_pages.json")
IMAGE_DIR = Path("data/processed/d4_images")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Remove noise
    denoised = cv2.medianBlur(gray, 3)

    # Improve contrast and create a binary image
    processed = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return processed


def extract_with_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    all_pages = []

    for page_number, page in enumerate(doc, start=1):
        print(f"\nProcessing page {page_number}...")

        # Render PDF page at 300 DPI
        pix = page.get_pixmap(
            matrix=fitz.Matrix(300 / 72, 300 / 72),
            alpha=False
        )

        image_path = IMAGE_DIR / f"d4_page_{page_number}.png"
        pix.save(image_path)

        # Read rendered image
        image = cv2.imread(str(image_path))

        if image is None:
            print("Could not load page image.")
            continue

        # Preprocess
        processed = preprocess_image(image)

        # Save preprocessed image
        processed_path = IMAGE_DIR / f"d4_page_{page_number}_processed.png"
        cv2.imwrite(str(processed_path), processed)

        # OCR
        text = pytesseract.image_to_string(
            processed,
            lang="eng",
            config="--oem 3 --psm 6"
        )

        all_pages.append({
            "document_id": "d4",
            "source_file": "d4.pdf",
            "page": page_number,
            "text": text.strip(),
            "character_count": len(text.strip()),
            "word_count": len(text.split()),
            "extraction_method": "tesseract_ocr"
        })

        print(f"Characters extracted: {len(text.strip())}")

    return all_pages


def main():
    if not PDF_FILE.exists():
        print(f"PDF not found: {PDF_FILE}")
        return

    pages = extract_with_ocr(PDF_FILE)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print("\nOCR completed.")
    print(f"Pages processed: {len(pages)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
