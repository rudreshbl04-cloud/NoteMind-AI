"""Unit and integration tests for PDF processor in NoteMind AI."""

import io
import unittest
from pypdf import PdfWriter
from src.pdf_processor import process_pdf, chunk_text, MAX_FILE_SIZE_BYTES


def create_sample_pdf(pages_text: list[str]) -> bytes:
    """Helper to generate a real PDF in-memory with given text per page."""
    writer = PdfWriter()
    for text in pages_text:
        # Create a page and add text
        page = writer.add_blank_page(width=612, height=792)
        # Note: blank page with no text stream will be empty.
        # To add actual text annotation or text content in pypdf:
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


class TestPdfProcessor(unittest.TestCase):
    """Test suite for PDF loading, extraction, chunking, and validation."""

    def test_chunking_text_basic(self):
        text = "Python is an interpreted, high-level, general-purpose programming language."
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        self.assertTrue(len(chunks) >= 2)
        # Verify no empty chunks
        for c in chunks:
            self.assertTrue(len(c.strip()) > 0)

    def test_chunking_empty_text(self):
        chunks = chunk_text("   \n\t  ")
        self.assertEqual(chunks, [])

    def test_empty_bytes_handling(self):
        chunks, err = process_pdf(b"", "empty.pdf")
        self.assertEqual(chunks, [])
        self.assertIn("empty", err.lower())

    def test_corrupted_pdf_handling(self):
        corrupted_bytes = b"%PDF-1.4 Not a real pdf content junk data"
        chunks, err = process_pdf(corrupted_bytes, "corrupt.pdf")
        self.assertEqual(chunks, [])
        self.assertTrue(err is not None)
        self.assertTrue("corrupted" in err.lower() or "error" in err.lower())

    def test_oversized_file_handling(self):
        dummy_large_bytes = b"0" * (MAX_FILE_SIZE_BYTES + 1024)
        chunks, err = process_pdf(dummy_large_bytes, "large.pdf")
        self.assertEqual(chunks, [])
        self.assertIn("exceeds the maximum allowed size", err)

    def test_blank_scanned_pdf_detection(self):
        # A blank PDF has pages but 0 extractable text characters
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        stream = io.BytesIO()
        writer.write(stream)
        blank_pdf_bytes = stream.getvalue()

        chunks, err = process_pdf(blank_pdf_bytes, "scanned_doc.pdf")
        self.assertEqual(chunks, [])
        self.assertIsNotNone(err)
        self.assertIn("scanned or image-only", err)

    def test_real_pdf_text_extraction_and_metadata(self):
        raw_pdf = b"""%PDF-1.4
1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj
2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj
3 0 obj <</Type /Page /Parent 2 0 R /Resources <</Font <</F1 4 0 R>>>> /MediaBox [0 0 612 792] /Contents 5 0 R>> endobj
4 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj
5 0 obj <</Length 44>> stream
BT
/F1 12 Tf
100 700 Td
(Python was created by Guido van Rossum.) Tj
ET
endstream endobj
xref
0 6
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000240 00000 n 
0000000318 00000 n 
trailer <</Size 6 /Root 1 0 R>>
startxref
413
%%EOF
"""
        chunks, err = process_pdf(raw_pdf, "lecture_notes.pdf")
        self.assertIsNone(err)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source"], "lecture_notes.pdf")
        self.assertEqual(chunks[0]["page"], 1)
        self.assertIn("Guido van Rossum", chunks[0]["text"])



if __name__ == "__main__":
    unittest.main()
