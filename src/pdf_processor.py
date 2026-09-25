"""PDF processing and text chunking for NoteMind AI.

Extracts text page-by-page from uploaded PDF files, generates chunks with
preserved source and page metadata, and detects scanned/empty/corrupted files.
"""

import io
from typing import Any, Dict, List, Optional, Tuple
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Configuration constants
CHUNK_SIZE = 1000  # Target character length per chunk (preserves complete paragraphs)
CHUNK_OVERLAP = 200  # Character overlap between adjacent chunks
MAX_FILE_SIZE_MB = 15  # Maximum allowed file size in megabytes
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks while preserving readability.

    Args:
        text: The input text to be segmented.
        chunk_size: Target character length per chunk.
        chunk_overlap: Overlapping characters between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    clean_text = " ".join(text.split()).strip()
    if not clean_text:
        return []

    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks = []
    start = 0
    text_length = len(clean_text)
    step = max(1, chunk_size - chunk_overlap)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start += step

    return chunks


def process_pdf(
    file_bytes: bytes,
    filename: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Process a single PDF from bytes and return extracted chunks with metadata.

    Args:
        file_bytes: Raw binary content of the PDF file.
        filename: Name of the uploaded PDF file.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        A tuple of (chunks_data, error_message):
        - chunks_data: List of dicts, each with keys 'text', 'source', 'page'
        - error_message: None if successful, or a friendly description if failed.
    """
    if not file_bytes:
        return [], f"The file '{filename}' is empty (0 bytes)."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return [], f"'{filename}' exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB."

    try:
        pdf_stream = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        if len(reader.pages) == 0:
            return [], f"'{filename}' contains no pages."

        extracted_chunks: List[Dict[str, Any]] = []
        total_extracted_characters = 0

        for page_index, page in enumerate(reader.pages):
            page_number = page_index + 1  # 1-indexed for human-friendly citations
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            page_text = page_text.strip()
            total_extracted_characters += len(page_text)

            if not page_text:
                continue

            page_chunks = chunk_text(page_text, chunk_size, chunk_overlap)
            for chunk in page_chunks:
                extracted_chunks.append({
                    "text": chunk,
                    "source": filename,
                    "page": page_number,
                })

        # Detection for scanned or empty PDFs
        if total_extracted_characters == 0:
            return (
                [],
                f"No readable text found in '{filename}'. This PDF appears to be scanned or image-only. "
                "OCR is not supported in Version 1.",
            )

        if not extracted_chunks:
            return [], f"Could not create text chunks from '{filename}'."

        return extracted_chunks, None

    except PdfReadError:
        return [], f"'{filename}' could not be read. The PDF file appears to be corrupted or invalid."
    except Exception as e:
        return [], f"Error reading '{filename}': {str(e)}"
