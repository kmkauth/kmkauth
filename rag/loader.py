"""
loader.py — Load PDFs and build a two-level hierarchy of chunks.

Hierarchy:
  Parent chunks (~1500 chars) — wide context, sent to Claude
  Child  chunks  (~300 chars) — narrow, precise, used for vector search

Each child stores its parent's ID so we can "expand up" at retrieval time.

GROBID integration:
  When GROBID is running locally (http://localhost:8070) each PDF is parsed
  into labelled sections with author/year/title metadata attached to every
  chunk.  If GROBID is unavailable the loader falls back silently to pypdf
  page-by-page extraction.
"""

import hashlib
from pathlib import Path
from pypdf import PdfReader

from rag.grobid import parse_pdf as grobid_parse


# ── Tuning knobs ─────────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE = 1500   # characters per parent
PARENT_OVERLAP    = 150    # overlap between parents
CHILD_CHUNK_SIZE  = 300    # characters per child  (must be < PARENT_CHUNK_SIZE)
CHILD_OVERLAP     = 30     # overlap between children within a parent
# ─────────────────────────────────────────────────────────────────────────────

# Metadata keys beyond {text, source, page} that flow through to ChromaDB
_EXTRA_META_KEYS = ("title", "authors", "year", "section")


def _stable_id(text: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(text.encode()).hexdigest()


def _split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size windows."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Per-file loaders ──────────────────────────────────────────────────────────

def _load_via_grobid(pdf_path: str) -> list[dict] | None:
    """
    Try to load structured sections from GROBID.

    Returns a list of page-like dicts — one per section — or None if GROBID
    is unavailable so the caller can fall back.
    """
    result = grobid_parse(pdf_path)
    if result is None or not result["sections"]:
        return None

    source = Path(pdf_path).name
    authors_str = ", ".join(result["authors"]) if result["authors"] else ""

    pages = []
    for idx, sec in enumerate(result["sections"], start=1):
        if sec["text"].strip():
            pages.append({
                "text":    sec["text"],
                "source":  source,
                "page":    idx,          # section index (GROBID has no page numbers)
                "section": sec["title"],
                "title":   result["title"],
                "authors": authors_str,
                "year":    result["year"] or "",
            })
    return pages or None


def _load_via_pypdf(pdf_path: str) -> list[dict]:
    """Fallback: extract text page-by-page with pypdf (no structured metadata)."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "text":    text,
                "source":  Path(pdf_path).name,
                "page":    i + 1,
                "section": "",
                "title":   "",
                "authors": "",
                "year":    "",
            })
    return pages


def load_pdf(pdf_path: str) -> list[dict]:
    """
    Return one dict per section (GROBID) or page (pypdf fallback).

    Every dict contains: text, source, page, section, title, authors, year
    """
    pages = _load_via_grobid(pdf_path)
    if pages is None:
        pages = _load_via_pypdf(pdf_path)
    return pages


# ── Hierarchy builder ─────────────────────────────────────────────────────────

def build_hierarchy(pages: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Build parent and child chunk lists from a list of page/section dicts.

    Returns:
        parents:  list of dicts with keys  id, text, source, page, section,
                                           title, authors, year
        children: list of dicts with keys  id, text, source, page, section,
                                           title, authors, year, parent_id
    """
    parents: list[dict] = []
    children: list[dict] = []

    for page in pages:
        extra = {k: page.get(k, "") for k in _EXTRA_META_KEYS}
        parent_texts = _split(page["text"], PARENT_CHUNK_SIZE, PARENT_OVERLAP)

        for parent_text in parent_texts:
            parent_id = _stable_id(parent_text, prefix="p_")
            parents.append({
                "id":     parent_id,
                "text":   parent_text,
                "source": page["source"],
                "page":   page["page"],
                **extra,
            })

            child_texts = _split(parent_text, CHILD_CHUNK_SIZE, CHILD_OVERLAP)
            for child_text in child_texts:
                child_id = _stable_id(child_text, prefix="c_")
                children.append({
                    "id":        child_id,
                    "text":      child_text,
                    "source":    page["source"],
                    "page":      page["page"],
                    "parent_id": parent_id,
                    **extra,
                })

    return parents, children


# ── Folder loader ─────────────────────────────────────────────────────────────

def load_pdfs_from_folder(folder: str = "rag/data/pdfs") -> tuple[list[dict], list[dict]]:
    """Load every PDF in folder and return (parents, children)."""
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

        via = "GROBID sections" if pages and pages[0].get("section") else "pypdf pages"
        print(f"    -> {len(pages)} {via} | {len(parents)} parents | {len(children)} children")

    return all_parents, all_children
