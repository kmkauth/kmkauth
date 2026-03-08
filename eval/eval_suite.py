"""
eval/eval_suite.py — Full evaluation harness for the hierarchical RAG system.

Runs every query in eval/queries.py against the live RAG system, measures:
  - Latency (seconds end-to-end including LLM call)
  - Keyword presence / absence checks
  - Source filter compliance (year/author filters respected?)
  - Hallucination detection (refusal check for HALLUCINATION category)
  - LLM-as-judge scores: relevance, faithfulness, completeness, citation
  - Per-category aggregates

Outputs:
  - eval/results/run_<timestamp>.json  — full machine-readable results
  - eval/results/run_<timestamp>.md    — white-paper-ready summary tables

Usage:
    python -m eval.eval_suite                   # run all queries
    python -m eval.eval_suite --category F      # run only FACTUAL queries
    python -m eval.eval_suite --id F-01         # run a single query by ID
    python -m eval.eval_suite --no-judge        # skip LLM-as-judge (faster)
    python -m eval.eval_suite --baseline        # compare against flat-RAG baseline

    # Run with baseline comparison:
    python -m eval.eval_suite --baseline

Prerequisites:
    ANTHROPIC_API_KEY must be set.
    PDFs must be ingested:  python main.py --ingest
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure the repo root is on the path when run as a module from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.query import ask_with_sources
from eval.queries import QUERIES
from eval.judge import judge_answer


# ---------------------------------------------------------------------------
# Optional baseline — flat RAG using pypdf page chunks, no hierarchy
# ---------------------------------------------------------------------------
def _ask_baseline(question: str, top_k: int, filter_year, filter_author) -> dict:
    """
    Thin shim: calls the same ask_with_sources but with top_k=1 to simulate
    flat retrieval. Replace this with a genuine flat-RAG implementation if
    you have one, to produce a fair ablation comparison.

    For a rigorous baseline, ingest into a separate flat ChromaDB collection
    using pypdf page splits at 512 chars, then query that collection here.
    """
    # Placeholder: real baseline should use a separate DB or different parameters.
    # For now, returns a clearly labelled stub.
    return {
        "answer": "[BASELINE NOT CONFIGURED — see eval_suite.py _ask_baseline]",
        "sources": [],
    }


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def _check_keywords(text: str, keywords: list[str]) -> dict:
    """Returns which expected keywords are present/absent in text (case-insensitive)."""
    text_lower = text.lower()
    found = [kw for kw in keywords if kw.lower() in text_lower]
    missing = [kw for kw in keywords if kw.lower() not in text_lower]
    return {"found": found, "missing": missing}


def _check_forbidden(text: str, forbidden: list[str]) -> list[str]:
    """Returns any forbidden keywords that appear in text (hallucination signal)."""
    text_lower = text.lower()
    return [kw for kw in forbidden if kw.lower() in text_lower]


def _check_refusal(text: str) -> bool:
    """
    Heuristic: does the answer acknowledge it cannot find the information?
    True = system appropriately refused / said it didn't know.
    """
    refusal_phrases = [
        "not in the context",
        "not found in",
        "cannot find",
        "don't have information",
        "do not have information",
        "no relevant document",
        "context doesn't contain",
        "context does not contain",
        "not enough information",
        "unable to find",
        "not mentioned",
        "not provided in",
        "i don't know",
        "i do not know",
        "cannot answer",
        "not available",
        "not present in",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in refusal_phrases)


def _check_source_filter(sources: list[dict], filter_year: str | None, filter_author: str | None) -> dict:
    """
    Validates that metadata filters were actually respected in the returned sources.
    Returns a compliance report.
    """
    if not sources:
        return {"year_compliant": None, "author_compliant": None, "violations": []}

    violations = []
    year_compliant = True
    author_compliant = True

    for src in sources:
        if filter_year and src.get("year") and src["year"] != filter_year:
            violations.append(f"Year violation: source year={src['year']} (expected {filter_year})")
            year_compliant = False
        if filter_author and src.get("authors"):
            if filter_author.lower() not in src["authors"].lower():
                violations.append(f"Author violation: source authors='{src['authors']}' (expected '{filter_author}')")
                author_compliant = False

    return {
        "year_compliant": year_compliant if filter_year else None,
        "author_compliant": author_compliant if filter_author else None,
        "violations": violations,
    }


def run_single(
    query: dict,
    use_judge: bool = True,
    run_baseline: bool = False,
) -> dict:
    """
    Run one test query. Returns a result record.
    """
    qid = query["id"]
    question = query["question"]

    print(f"  [{qid}] {question[:80]}{'...' if len(question) > 80 else ''}")

    # ----- Primary system -----
    t0 = time.perf_counter()
    try:
        result = ask_with_sources(
            question=question,
            top_k=query.get("top_k", 5),
            filter_year=query.get("filter_year"),
            filter_author=query.get("filter_author"),
        )
        latency = time.perf_counter() - t0
        answer = result["answer"]
        sources = result["sources"]
        error = None
    except Exception as exc:
        latency = time.perf_counter() - t0
        answer = ""
        sources = []
        error = str(exc)
        print(f"    ERROR: {error}")

    # ----- Automated checks -----
    keyword_check = _check_keywords(answer, query.get("expected_keywords", []))
    forbidden_hits = _check_forbidden(answer, query.get("forbidden_keywords", []))
    source_filter_check = _check_source_filter(
        sources,
        query.get("filter_year"),
        query.get("filter_author"),
    )
    refused = _check_refusal(answer)
    expect_refusal = query.get("expect_refusal", False)
    refusal_correct = (refused == expect_refusal)   # True = system behaved as expected

    # Source contains check
    expected_source_contains = query.get("expected_source_contains", [])
    source_texts = " ".join(
        f"{s.get('source','')} {s.get('authors','')}" for s in sources
    ).lower()
    source_contains_hits = [t for t in expected_source_contains if t.lower() in source_texts]
    source_contains_missing = [t for t in expected_source_contains if t.lower() not in source_texts]

    # ----- LLM-as-judge -----
    judge_scores = None
    if use_judge and answer and not error:
        try:
            judge_scores = judge_answer(
                question=question,
                answer=answer,
                context_chunks=sources,
                ground_truth=query.get("ground_truth", ""),
            )
        except Exception as exc:
            print(f"    JUDGE ERROR: {exc}")
            judge_scores = {"relevance": 0, "faithfulness": 0, "completeness": 0, "citation": 0, "total": 0, "reasoning": str(exc)}

    # ----- Baseline (optional) -----
    baseline_result = None
    baseline_latency = None
    if run_baseline:
        tb = time.perf_counter()
        try:
            baseline_result = _ask_baseline(
                question,
                top_k=query.get("top_k", 5),
                filter_year=query.get("filter_year"),
                filter_author=query.get("filter_author"),
            )
        except Exception:
            baseline_result = {"answer": "", "sources": []}
        baseline_latency = time.perf_counter() - tb

    record = {
        "id": qid,
        "category": query["category"],
        "question": question,
        "filters": {
            "year": query.get("filter_year"),
            "author": query.get("filter_author"),
        },
        "latency_s": round(latency, 3),
        "answer": answer,
        "num_sources": len(sources),
        "sources": [
            {
                "file": s.get("source", ""),
                "authors": s.get("authors", ""),
                "year": s.get("year", ""),
                "section": s.get("section", ""),
                "score": s.get("score", 0.0),
            }
            for s in sources
        ],
        "checks": {
            "keyword_found": keyword_check["found"],
            "keyword_missing": keyword_check["missing"],
            "keyword_pass": len(keyword_check["missing"]) == 0,
            "forbidden_hits": forbidden_hits,
            "forbidden_pass": len(forbidden_hits) == 0,
            "refusal_expected": expect_refusal,
            "refusal_detected": refused,
            "refusal_correct": refusal_correct,
            "source_filter": source_filter_check,
            "source_contains_hits": source_contains_hits,
            "source_contains_missing": source_contains_missing,
        },
        "judge": judge_scores,
        "baseline_latency_s": round(baseline_latency, 3) if baseline_latency is not None else None,
        "baseline_answer": baseline_result["answer"] if baseline_result else None,
        "error": error,
        "notes": query.get("notes", ""),
    }

    # Print quick summary line
    status = "PASS" if (
        record["checks"]["keyword_pass"] and
        record["checks"]["forbidden_pass"] and
        record["checks"]["refusal_correct"]
    ) else "FAIL"
    judge_str = f"  judge={judge_scores['total']}/20" if judge_scores else ""
    print(f"    {status} | latency={latency:.2f}s | sources={len(sources)}{judge_str}")

    return record


# ---------------------------------------------------------------------------
# Aggregate and report
# ---------------------------------------------------------------------------

def _category_stats(results: list[dict]) -> dict:
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    stats = {}
    for cat, items in by_cat.items():
        valid = [i for i in items if not i["error"]]
        judge_totals = [i["judge"]["total"] for i in valid if i.get("judge")]
        latencies = [i["latency_s"] for i in valid]
        keyword_passes = [i["checks"]["keyword_pass"] for i in valid if i["checks"]["keyword_found"] or i["checks"]["keyword_missing"]]
        refusal_corrects = [i["checks"]["refusal_correct"] for i in valid if i["checks"]["refusal_expected"]]

        stats[cat] = {
            "n": len(items),
            "errors": len(items) - len(valid),
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "max_latency_s": round(max(latencies), 3) if latencies else None,
            "avg_judge_total": round(sum(judge_totals) / len(judge_totals), 1) if judge_totals else None,
            "avg_relevance":    round(sum(i["judge"]["relevance"] for i in valid if i.get("judge")) / len(judge_totals), 1) if judge_totals else None,
            "avg_faithfulness": round(sum(i["judge"]["faithfulness"] for i in valid if i.get("judge")) / len(judge_totals), 1) if judge_totals else None,
            "avg_completeness": round(sum(i["judge"]["completeness"] for i in valid if i.get("judge")) / len(judge_totals), 1) if judge_totals else None,
            "avg_citation":     round(sum(i["judge"]["citation"] for i in valid if i.get("judge")) / len(judge_totals), 1) if judge_totals else None,
            "keyword_pass_rate": f"{sum(keyword_passes)}/{len(keyword_passes)}" if keyword_passes else "N/A (no keywords set)",
            "hallucination_refusal_rate": f"{sum(refusal_corrects)}/{len(refusal_corrects)}" if refusal_corrects else "N/A",
        }
    return stats


def _write_markdown(results: list[dict], stats: dict, out_path: Path, run_id: str) -> None:
    lines = [
        f"# RAG Evaluation Report",
        f"",
        f"**Run ID:** `{run_id}`  ",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Total queries:** {len(results)}  ",
        f"**Queries with errors:** {sum(1 for r in results if r['error'])}",
        f"",
        f"---",
        f"",
        f"## Summary by Category",
        f"",
        f"| Category | N | Avg Latency (s) | Avg Score /20 | Faithfulness /5 | Refusal Accuracy |",
        f"|---|---|---|---|---|---|",
    ]
    for cat, s in stats.items():
        lines.append(
            f"| {cat} | {s['n']} | {s['avg_latency_s']} | {s['avg_judge_total'] or 'N/A'} "
            f"| {s['avg_faithfulness'] or 'N/A'} | {s['hallucination_refusal_rate']} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Score Breakdown by Category",
        f"",
        f"| Category | Relevance /5 | Faithfulness /5 | Completeness /5 | Citation /5 | Total /20 |",
        f"|---|---|---|---|---|---|",
    ]
    for cat, s in stats.items():
        lines.append(
            f"| {cat} | {s['avg_relevance'] or '-'} | {s['avg_faithfulness'] or '-'} "
            f"| {s['avg_completeness'] or '-'} | {s['avg_citation'] or '-'} | {s['avg_judge_total'] or '-'} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Per-Query Results",
        f"",
    ]

    current_cat = None
    for r in results:
        if r["category"] != current_cat:
            current_cat = r["category"]
            lines += [f"### {current_cat}", f""]

        status_emoji = "✓" if (
            r["checks"]["keyword_pass"] and
            r["checks"]["forbidden_pass"] and
            r["checks"]["refusal_correct"]
        ) else "✗"

        lines += [
            f"#### [{r['id']}] {status_emoji} {r['question']}",
            f"",
            f"- **Latency:** {r['latency_s']}s",
            f"- **Sources retrieved:** {r['num_sources']}",
        ]

        if r["filters"]["year"] or r["filters"]["author"]:
            lines.append(f"- **Filters applied:** year={r['filters']['year']} author={r['filters']['author']}")

        if r["checks"]["source_filter"]["violations"]:
            lines.append(f"- **FILTER VIOLATION:** {'; '.join(r['checks']['source_filter']['violations'])}")

        if r["checks"]["keyword_missing"]:
            lines.append(f"- **Missing expected keywords:** {r['checks']['keyword_missing']}")

        if r["checks"]["forbidden_hits"]:
            lines.append(f"- **HALLUCINATION SIGNAL — forbidden keywords found:** {r['checks']['forbidden_hits']}")

        if r["checks"]["refusal_expected"] and not r["checks"]["refusal_correct"]:
            lines.append(f"- **HALLUCINATION: system should have refused but did not**")

        if r.get("judge"):
            j = r["judge"]
            lines.append(
                f"- **Judge scores:** Relevance={j['relevance']} Faithfulness={j['faithfulness']} "
                f"Completeness={j['completeness']} Citation={j['citation']} **Total={j['total']}/20**"
            )
            lines.append(f"- **Judge reasoning:** _{j.get('reasoning', '')}_ ")

        lines += [
            f"",
            f"**Answer:**",
            f"> {r['answer'][:500].replace(chr(10), ' ')}{'...' if len(r['answer']) > 500 else ''}",
            f"",
            f"**Sources:**",
        ]
        for src in r["sources"]:
            lines.append(
                f"- `{src['file']}` | {src['authors']} ({src['year']}) | §{src['section']} | score={src['score']:.4f}"
            )

        if r.get("notes"):
            lines.append(f"")
            lines.append(f"*Test intent: {r['notes']}*")

        lines.append(f"")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation suite")
    parser.add_argument("--category", metavar="CAT",
                        help="Run only queries in this category (e.g. F, FACTUAL, H)")
    parser.add_argument("--id", metavar="ID",
                        help="Run a single query by ID (e.g. F-01)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM-as-judge scoring (faster, cheaper)")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline comparison alongside each query")
    parser.add_argument("--out-dir", default="eval/results",
                        help="Output directory for reports (default: eval/results)")
    args = parser.parse_args()

    # Filter queries
    queries = QUERIES
    if args.id:
        queries = [q for q in QUERIES if q["id"] == args.id]
        if not queries:
            print(f"No query with id '{args.id}'")
            sys.exit(1)
    elif args.category:
        cat_upper = args.category.upper()
        queries = [q for q in QUERIES if q["category"].startswith(cat_upper)]
        if not queries:
            print(f"No queries in category '{args.category}'")
            sys.exit(1)

    # Check ingestion
    from rag.vectorstore import collection_sizes
    sizes = collection_sizes()
    if sizes["children"] == 0:
        print("WARNING: No documents ingested. Run: python main.py --ingest")
        print("Continuing anyway — all queries will return no-source refusals.\n")

    print(f"\n{'='*60}")
    print(f"RAG Evaluation Suite")
    print(f"{'='*60}")
    print(f"Queries to run : {len(queries)}")
    print(f"LLM-as-judge   : {'disabled' if args.no_judge else 'enabled'}")
    print(f"Baseline        : {'enabled' if args.baseline else 'disabled'}")
    print(f"Vector store    : {sizes['children']} child chunks, {sizes['parents']} parent chunks")
    print(f"{'='*60}\n")

    results = []
    by_category: dict[str, list] = {}
    for q in queries:
        by_category.setdefault(q["category"], []).append(q)

    for cat, cat_queries in by_category.items():
        print(f"\n--- Category: {cat} ({len(cat_queries)} queries) ---")
        for q in cat_queries:
            r = run_single(q, use_judge=not args.no_judge, run_baseline=args.baseline)
            results.append(r)

    # Aggregate
    stats = _category_stats(results)

    # Output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"run_{run_id}.json"
    md_path = out_dir / f"run_{run_id}.md"

    json_path.write_text(json.dumps({"run_id": run_id, "stats": stats, "results": results}, indent=2))
    _write_markdown(results, stats, md_path, run_id)

    # Print final summary
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nCategory Averages:")
    for cat, s in stats.items():
        print(f"  {cat:20s}  latency={s['avg_latency_s']}s  judge={s['avg_judge_total'] or 'N/A'}/20  faithfulness={s['avg_faithfulness'] or 'N/A'}/5")

    overall_judge = [r["judge"]["total"] for r in results if r.get("judge")]
    if overall_judge:
        print(f"\nOverall average judge score: {sum(overall_judge)/len(overall_judge):.1f} / 20")

    hallucination_results = [r for r in results if r["category"] == "HALLUCINATION"]
    if hallucination_results:
        correct_refusals = sum(1 for r in hallucination_results if r["checks"]["refusal_correct"])
        print(f"Hallucination refusal accuracy: {correct_refusals}/{len(hallucination_results)}")

    forbidden_hits = [r for r in results if r["checks"]["forbidden_hits"]]
    if forbidden_hits:
        print(f"Queries with forbidden keywords in answer: {len(forbidden_hits)}")
        for r in forbidden_hits:
            print(f"  [{r['id']}] {r['checks']['forbidden_hits']}")

    print(f"\nReports saved:")
    print(f"  JSON : {json_path}")
    print(f"  Markdown : {md_path}")
    print(f"\nThe markdown report can be pasted directly into your white paper §5 Evaluation section.")


if __name__ == "__main__":
    main()
