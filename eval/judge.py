"""
eval/judge.py — LLM-as-judge scoring for RAG answer quality.

Uses Claude itself as an impartial evaluator (a common, well-validated approach
in RAG evaluation literature). Each answer is scored on four dimensions:

  1. Relevance    (0-5)  Does the answer address the question?
  2. Faithfulness (0-5)  Is every claim grounded in the provided sources?
                          (Key hallucination test — no invented facts)
  3. Completeness (0-5)  Does the answer cover all parts of the question?
  4. Citation     (0-5)  Does the answer cite specific authors and years
                          rather than vague 'the document' references?

Total score: 0–20. Each dimension is scored independently to give a breakdown
that can be reported per category in the white paper (Table 1 style).

For HALLUCINATION category queries, Faithfulness carries the most weight:
a confabulated answer on an off-topic query scores 0 on Faithfulness
regardless of how polished it sounds.
"""

import json
import os
import anthropic


_JUDGE_PROMPT = """\
You are an expert evaluator assessing the quality of answers produced by a \
Retrieval-Augmented Generation (RAG) system. Score the answer on four dimensions. \
Return ONLY a JSON object — no prose, no markdown fences.

QUESTION:
{question}

CONTEXT PROVIDED TO THE SYSTEM (the only information it should use):
{context}

SYSTEM ANSWER:
{answer}

GROUND TRUTH (may be empty — use as reference only if provided):
{ground_truth}

Score each dimension from 0 to 5 using these rubrics:

RELEVANCE (0-5):
  5 = Directly and completely addresses the question
  3 = Partially addresses the question or goes off-topic in parts
  1 = Barely addresses the question
  0 = Completely off-topic or refuses without explanation

FAITHFULNESS (0-5):
  5 = Every claim is directly supported by the context; no invented facts
  3 = Mostly faithful; minor unsupported claims
  1 = Several claims not found in context (hallucination present)
  0 = Answer invents facts not in context, or states things contradicted by context

COMPLETENESS (0-5):
  5 = Covers all aspects of the question that the context allows
  3 = Covers the main point but misses supporting details
  1 = Only a superficial or partial answer
  0 = Effectively empty or says 'I don't know' when context has the answer

CITATION (0-5):
  5 = Cites specific author names and/or years for all key claims
  3 = Cites some sources but inconsistently
  1 = Vague references ('the document', 'the paper') without names
  0 = No attribution at all, or citations are fabricated

Return exactly this JSON structure:
{{
  "relevance":     <int 0-5>,
  "faithfulness":  <int 0-5>,
  "completeness":  <int 0-5>,
  "citation":      <int 0-5>,
  "reasoning":     "<one sentence explaining the key strengths/weaknesses>"
}}
"""


def judge_answer(
    question: str,
    answer: str,
    context_chunks: list[dict],
    ground_truth: str = "",
    model: str = "claude-sonnet-4-6",
) -> dict:
    """
    Run LLM-as-judge scoring on a single RAG answer.

    Args:
        question:       The original question.
        answer:         The RAG system's answer string.
        context_chunks: The list of source hit dicts returned by ask_with_sources().
        ground_truth:   Optional gold reference answer.
        model:          Claude model to use as judge.

    Returns:
        {
          "relevance":    int,
          "faithfulness": int,
          "completeness": int,
          "citation":     int,
          "total":        int,   # sum of the four scores (max 20)
          "reasoning":    str,
        }
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")

    # Build a compact representation of the context the system had available
    context_lines = []
    for i, hit in enumerate(context_chunks, 1):
        citation = f"{hit.get('authors', '')} ({hit.get('year', '')}) §{hit.get('section', '')}"
        context_lines.append(f"[Source {i}: {citation}]\n{hit['text'][:600]}")
    context_text = "\n\n---\n\n".join(context_lines) if context_lines else "(no sources retrieved)"

    prompt = _JUDGE_PROMPT.format(
        question=question,
        context=context_text,
        answer=answer,
        ground_truth=ground_truth or "(not provided)",
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return zeros with the raw text as reasoning
        return {
            "relevance": 0, "faithfulness": 0, "completeness": 0, "citation": 0,
            "total": 0, "reasoning": f"Judge parse error: {raw[:200]}",
        }

    scores["total"] = (
        scores.get("relevance", 0) +
        scores.get("faithfulness", 0) +
        scores.get("completeness", 0) +
        scores.get("citation", 0)
    )
    return scores
