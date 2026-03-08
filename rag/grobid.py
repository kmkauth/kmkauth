"""
grobid.py — Extract structured metadata and full-text sections from PDFs via GROBID.

GROBID must be running locally.  Quickest way with Docker:
    docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0

Or native install: https://grobid.readthedocs.io/en/latest/Install-Grobid/

If GROBID is unavailable the functions return None and the loader falls back to pypdf.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Optional

import requests

GROBID_URL = "http://localhost:8070"
TEI_NS = "http://www.tei-c.org/ns/1.0"

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tag(local: str) -> str:
    return f"{{{TEI_NS}}}{local}"


def _text(el) -> str:
    """Recursively collect all text inside an element."""
    return " ".join(el.itertext()).strip()


def _find(root, *local_tags):
    """Walk a chain of local tag names, returning the element or None."""
    el = root
    for tag in local_tags:
        if el is None:
            return None
        el = el.find(_tag(tag))
    return el


# ── Public API ────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> Optional[dict]:
    """
    Send a PDF to GROBID's processFulltextDocument endpoint and parse the result.

    Returns a dict on success:
        {
            "title":    str,
            "authors":  list[str],          # e.g. ["Smith, J.", "Jones, A."]
            "year":     str | None,         # e.g. "2019"
            "sections": [
                {"title": str, "text": str},
                ...
            ]
        }

    Returns None if GROBID is unreachable or parsing fails (caller should fall back).
    """
    try:
        with open(pdf_path, "rb") as fh:
            resp = requests.post(
                f"{GROBID_URL}/api/processFulltextDocument",
                files={"input": fh},
                data={"consolidateHeader": "1"},
                timeout=60,
            )
        resp.raise_for_status()
    except Exception as exc:
        log.warning("GROBID unavailable (%s) — will fall back to pypdf for: %s", exc, pdf_path)
        return None

    try:
        return _parse_tei(resp.text)
    except Exception as exc:
        log.warning("GROBID TEI parse error (%s) — will fall back to pypdf for: %s", exc, pdf_path)
        return None


# ── TEI XML parser ────────────────────────────────────────────────────────────

def _parse_tei(tei_xml: str) -> dict:
    root = ET.fromstring(tei_xml)

    header = root.find(f".//{_tag('teiHeader')}")

    # Title — prefer level="a" (analytic / article), fall back to any <title>
    title_el = None
    if header is not None:
        title_el = header.find(f".//{_tag('title')}[@level='a']")
        if title_el is None:
            title_el = header.find(f".//{_tag('title')}")
    title = _text(title_el) if title_el is not None else ""

    # Authors
    authors: list[str] = []
    if header is not None:
        for author_el in header.findall(f".//{_tag('author')}"):
            forename_el = author_el.find(f".//{_tag('forename')}")
            surname_el  = author_el.find(f".//{_tag('surname')}")
            if surname_el is not None:
                name = _text(surname_el)
                if forename_el is not None:
                    initial = _text(forename_el)[:1]
                    name = f"{name}, {initial}."
                authors.append(name)

    # Publication year
    year: Optional[str] = None
    if header is not None:
        date_el = header.find(f".//{_tag('date')}[@type='published']")
        if date_el is None:
            date_el = header.find(f".//{_tag('date')}")
        if date_el is not None:
            when = date_el.get("when", "")
            year = when[:4] if len(when) >= 4 else (_text(date_el)[:4] or None)

    # Body sections — each <div> in <body> is one section
    body = root.find(f".//{_tag('body')}")
    sections: list[dict] = []
    if body is not None:
        for div in body.findall(_tag("div")):
            head_el = div.find(_tag("head"))
            section_title = _text(head_el) if head_el is not None else "Section"

            # Collect paragraph text (direct <p> children only, not nested divs)
            paragraphs = [_text(p) for p in div.findall(_tag("p")) if _text(p)]
            section_text = "\n\n".join(paragraphs)

            if section_text.strip():
                sections.append({"title": section_title, "text": section_text})

    return {
        "title":    title,
        "authors":  authors,
        "year":     year,
        "sections": sections,
    }
