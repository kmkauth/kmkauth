"""
query.py — Ask Claude a question using hierarchically retrieved context.

The key difference from flat RAG:
  - We search against small child chunks (precise matching)
  - We send large parent chunks to Claude (rich context)
  - We also show which child snippet triggered each parent (for transparency)
"""

import os
import anthropic
from rag.vectorstore import search


def ask_with_sources(
    question: str,
    top_k: int = 5,
    model: str = "claude-sonnet-4-6",
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
                    "score":         float, # cosine similarity of child match
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

    hits = search(question, top_k=top_k)
    if not hits:
        return {
            "answer": (
                "No relevant documents found. "
                "Have you ingested any PDFs yet? Run with --ingest first."
            ),
            "sources": [],
        }

    # Build context from PARENT chunks (large, rich context)
    context_parts = []
    for i, hit in enumerate(hits, 1):
        context_parts.append(
            f"[Source {i}: {hit['source']}, page {hit['page']} | relevance {hit['score']}]\n"
            f"{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the context "
        "provided below. If the context doesn't contain enough information, say so clearly.\n\n"
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


def ask(question: str, top_k: int = 5, model: str = "claude-sonnet-4-6") -> str:
    """Convenience wrapper — returns just the answer string."""
    return ask_with_sources(question, top_k=top_k, model=model)["answer"]
