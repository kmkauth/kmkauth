"""
query.py — Ask Claude a question using hierarchically retrieved context.

The key difference from flat RAG:
  - We search against small child chunks (precise matching)
  - We send large parent chunks to Claude (rich context)
  - GROBID metadata (author, year, section) is surfaced in both the context
    prompt and the returned source list.
"""

import os
import anthropic
from rag.vectorstore import search


def _citation(hit: dict) -> str:
    """Build a human-readable citation string from a hit's metadata."""
    parts = []
    if hit.get("authors"):
        parts.append(hit["authors"])
    if hit.get("year"):
        parts.append(f"({hit['year']})")
    label = " ".join(parts) if parts else hit.get("source", "Unknown")
    if hit.get("title"):
        label += f' — "{hit["title"]}"'
    if hit.get("section"):
        label += f"  §{hit['section']}"
    return label


def ask_with_sources(
    question: str,
    top_k: int = 5,
    model: str = "claude-sonnet-4-6",
    filter_year: str | None = None,
    filter_author: str | None = None,
) -> dict:
    """
    Full hierarchical RAG pipeline.

    Returns:
        {
            "answer":  str,
            "sources": [
                {
                    "text":          str,   # full parent chunk sent to Claude
                    "matched_child": str,   # the small snippet that matched the query
                    "source":        str,   # filename
                    "page":          str,
                    "section":       str,
                    "title":         str,
                    "authors":       str,
                    "year":          str,
                    "score":         float,
                }
            ]
        }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Export it with:  export ANTHROPIC_API_KEY=your_key_here"
        )

    hits = search(question, top_k=top_k, filter_year=filter_year, filter_author=filter_author)
    if not hits:
        return {
            "answer": (
                "No relevant documents found. "
                "Have you ingested any PDFs yet? Run with --ingest first."
            ),
            "sources": [],
        }

    # Build context from PARENT chunks — include rich citation metadata
    context_parts = []
    for i, hit in enumerate(hits, 1):
        citation = _citation(hit)
        context_parts.append(
            f"[Source {i}: {citation} | file: {hit['source']} | relevance {hit['score']}]\n"
            f"{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant for technical and scientific literature. "
        "Answer the user's question using ONLY the context provided below. "
        "When referring to sources, cite author names and years where available. "
        "If the context doesn't contain enough information, say so clearly.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer":  message.content[0].text,
        "sources": hits,
    }


def ask(
    question: str,
    top_k: int = 5,
    model: str = "claude-sonnet-4-6",
    filter_year: str | None = None,
    filter_author: str | None = None,
) -> str:
    """Convenience wrapper — returns just the answer string."""
    return ask_with_sources(
        question, top_k=top_k, model=model,
        filter_year=filter_year, filter_author=filter_author,
    )["answer"]
