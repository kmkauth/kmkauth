"""
query.py — Send retrieved context + user question to Claude and return an answer.
"""

import os
import anthropic
from rag.vectorstore import search


def ask(question: str, top_k: int = 5, model: str = "claude-sonnet-4-6") -> str:
    """
    Full RAG pipeline:
      1. Retrieve the most relevant chunks for the question.
      2. Build a prompt with those chunks as context.
      3. Send to Claude and return the answer.

    Requires the ANTHROPIC_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it with: export ANTHROPIC_API_KEY=your_key_here"
        )

    # Step 1: Retrieve relevant chunks
    hits = search(question, top_k=top_k)
    if not hits:
        return "No relevant documents found. Have you ingested any PDFs yet? Run with --ingest first."

    # Step 2: Build context block
    context_parts = []
    for i, hit in enumerate(hits, 1):
        context_parts.append(
            f"[Source {i}: {hit['source']}, page {hit['page']} | relevance {hit['score']}]\n{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say so clearly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    # Step 3: Call Claude
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def ask_with_sources(question: str, top_k: int = 5, model: str = "claude-sonnet-4-6") -> dict:
    """
    Same as ask() but also returns the source chunks used.

    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "page": str, "score": float, "text": str}, ...]
        }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it with: export ANTHROPIC_API_KEY=your_key_here"
        )

    hits = search(question, top_k=top_k)
    if not hits:
        return {
            "answer": "No relevant documents found. Have you ingested any PDFs yet? Run with --ingest first.",
            "sources": [],
        }

    context_parts = []
    for i, hit in enumerate(hits, 1):
        context_parts.append(
            f"[Source {i}: {hit['source']}, page {hit['page']} | relevance {hit['score']}]\n{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say so clearly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": message.content[0].text,
        "sources": hits,
    }
