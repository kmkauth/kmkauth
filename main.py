"""
main.py — CLI for the hierarchical RAG framework with GROBID metadata support.

Usage:
    # Step 1: ensure GROBID is running (optional but recommended for tech papers)
    #   docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0

    # Step 2: drop PDFs into rag/data/pdfs/

    # Step 3: ingest (builds parent + child chunks, embeds children)
    python main.py --ingest

    # Step 4: ask questions
    python main.py --ask "What are the key findings?"

    # Show matched snippets, section names, authors and years
    python main.py --ask "Explain the methodology" --show-sources

    # Filter to a specific year or author
    python main.py --ask "describe the architecture" --filter-year 2021
    python main.py --ask "loss function details"     --filter-author "Smith"

    # Check storage stats
    python main.py --status

    # Re-ingest from scratch (wipes existing collections first)
    python main.py --clear && python main.py --ingest
"""

import argparse
import sys
from rag.loader import load_pdfs_from_folder
from rag.vectorstore import add_parents, add_children, collection_sizes, clear_all
from rag.query import ask_with_sources


def ingest(folder: str) -> None:
    print(f"\nIngesting PDFs from '{folder}' using hierarchical chunking...")
    parents, children = load_pdfs_from_folder(folder)
    if not parents:
        return
    add_parents(parents)
    add_children(children)
    sizes = collection_sizes()
    print(
        f"\nDone!\n"
        f"  {sizes['parents']} parent chunks stored  (sent to Claude as context)\n"
        f"  {sizes['children']} child chunks embedded (searched against your queries)\n"
    )


def query(
    question: str,
    show_sources: bool,
    top_k: int,
    filter_year: str | None,
    filter_author: str | None,
) -> None:
    print(f"\nSearching for: {question!r}\n")
    result = ask_with_sources(
        question,
        top_k=top_k,
        filter_year=filter_year,
        filter_author=filter_author,
    )

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    if show_sources:
        print("\n" + "=" * 60)
        print("HOW IT WORKED  (hierarchical retrieval)")
        print("=" * 60)
        for i, src in enumerate(result["sources"], 1):
            # Build a rich header line
            header_parts = []
            if src.get("authors"):
                header_parts.append(src["authors"])
            if src.get("year"):
                header_parts.append(f"({src['year']})")
            if src.get("title"):
                header_parts.append(f'"{src["title"]}"')
            if not header_parts:
                header_parts.append(src["source"])

            loc_parts = [f"file: {src['source']}"]
            if src.get("section"):
                loc_parts.append(f"§{src['section']}")
            loc_parts.append(f"page/section {src['page']}")
            loc_parts.append(f"score {src['score']}")

            print(f"\n[{i}] {' '.join(header_parts)}")
            print(f"     {' | '.join(loc_parts)}")
            print(f"\n  Child snippet that matched your query:")
            snippet = src["matched_child"]
            print(f"    \"{snippet[:200]}{'...' if len(snippet) > 200 else ''}\"")
            print(f"\n  Parent chunk sent to Claude (full context):")
            context = src["text"]
            print(f"    \"{context[:300]}{'...' if len(context) > 300 else ''}\"")


def status() -> None:
    sizes = collection_sizes()
    print(
        f"\nVector store stats:\n"
        f"  Parent chunks : {sizes['parents']}\n"
        f"  Child chunks  : {sizes['children']}\n"
    )
    if sizes["children"] == 0:
        print("Tip: run  python main.py --ingest  after dropping PDFs into rag/data/pdfs/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical RAG — query your PDFs with Claude + GROBID metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ingest", action="store_true",
                        help="Load PDFs and store hierarchical chunks")
    parser.add_argument("--ask", metavar="QUESTION",
                        help="Ask a question about your documents")
    parser.add_argument("--show-sources", action="store_true",
                        help="Show matched child + parent chunks with full citation info")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of parent chunks to retrieve (default: 5)")
    parser.add_argument("--folder", default="rag/data/pdfs",
                        help="Folder containing PDFs (default: rag/data/pdfs)")
    parser.add_argument("--filter-year", metavar="YEAR",
                        help="Only retrieve chunks from this publication year (e.g. 2021)")
    parser.add_argument("--filter-author", metavar="NAME",
                        help="Only retrieve chunks from papers by this author (substring match)")
    parser.add_argument("--status", action="store_true",
                        help="Show storage stats")
    parser.add_argument("--clear", action="store_true",
                        help="Wipe all stored chunks (use before re-ingesting)")

    args = parser.parse_args()

    if args.clear:
        clear_all()
    elif args.ingest:
        ingest(args.folder)
    elif args.ask:
        query(
            args.ask,
            show_sources=args.show_sources,
            top_k=args.top_k,
            filter_year=args.filter_year,
            filter_author=args.filter_author,
        )
    elif args.status:
        status()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
