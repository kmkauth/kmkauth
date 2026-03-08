"""
loader.py — Load PDFs and split them into text chunks.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Read a PDF and return a list of pages as dicts with text and metadata.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({
                "text": text,
                "source": Path(pdf_path).name,
                "page": i + 1,
            })
    return pages


def chunk_text(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split page text into smaller overlapping chunks for better retrieval.

    Args:
        pages:      Output from load_pdf().
        chunk_size: Approximate number of characters per chunk.
        overlap:    Number of characters to overlap between consecutive chunks.
    """
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_start": start,
                })
            start += chunk_size - overlap
    return chunks


def load_pdfs_from_folder(folder: str = "rag/data/pdfs") -> list[dict]:
    """
    Convenience function: load and chunk every PDF in a folder.
    """
    folder_path = Path(folder)
    all_chunks = []
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{folder}'. Drop some PDFs there and try again.")
        return []
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        pages = load_pdf(str(pdf_file))
        chunks = chunk_text(pages)
        all_chunks.extend(chunks)
        print(f"    -> {len(chunks)} chunks from {len(pages)} pages")
    return all_chunks
