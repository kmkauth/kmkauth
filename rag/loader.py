"""
loader.py — Load PDFs and build a two-level hierarchy of chunks.

Hierarchy:
  Parent chunks (~1500 chars) — wide context, sent to Claude
  Child  chunks  (~300 chars) — narrow, precise, used for vector search

Each child stores its parent's ID so we can "expand up" at retrieval time.
"""

import hashlib
from pathlib import Path
from pypdf import PdfReader


# ── Tuning knobs ─────────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE = 1500   # characters per parent
PARENT_OVERLAP    = 150    # overlap between parents
CHILD_CHUNK_SIZE  = 300    # characters per child  (must be < PARENT_CHUNK_SIZE)
CHILD_OVERLAP     = 30     # overlap between children within a parent
# ─────────────────────────────────────────────────────────────────────────────


def _stable_id(text: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(text.encode()).hexdigest()


def load_pdf(pdf_path: str) -> list[dict]:
    """Return one dict per page: {text, source, page}."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "source": Path(pdf_path).name, "page": i + 1})
    return pages


def _split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_hierarchy(pages: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Build parent and child chunk lists from a list of pages.

    Returns:
        parents: list of dicts with keys  id, text, source, page
        children: list of dicts with keys id, text, source, page, parent_id
    """
    parents: list[dict] = []
    children: list[dict] = []

    for page in pages:
        parent_texts = _split(page["text"], PARENT_CHUNK_SIZE, PARENT_OVERLAP)

        for parent_text in parent_texts:
            parent_id = _stable_id(parent_text, prefix="p_")
            parents.append({
                "id":     parent_id,
                "text":   parent_text,
                "source": page["source"],
                "page":   page["page"],
            })

            # Slice this parent into smaller children
            child_texts = _split(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)
            for child_text in child_texts:
                child_id = _stable_id(child_text, prefix="c_")
                children.append({
                    "id":        child_id,
                    "text":      child_text,
                    "source":    page["source"],
                    "page":      page["page"],
                    "parent_id": parent_id,
                })

    return parents, children


def load_pdfs_from_folder(folder: str = "rag/data/pdfs") -> tuple[list[dict], list[dict]]:
    """
    Load every PDF in folder and return (parents, children).
    """
    folder_path = Path(folder)
    all_parents: list[dict] = []
    all_children: list[dict] = []

    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{folder}'. Drop some PDFs there and run --ingest.")
        return [], []

    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file.name}")
        pages = load_pdf(str(pdf_file))
        parents, children = build_hierarchy(pages)
        all_parents.extend(parents)
        all_children.extend(children)
        print(f"    -> {len(pages)} pages | {len(parents)} parents | {len(children)} children")

    return all_parents, all_children
