"""
main.py — CLI for the hierarchical RAG framework.

Usage:
    # Step 1: drop PDFs into rag/data/pdfs/

    # Step 2: ingest (builds parent + child chunks, embeds children)
    python main.py --ingest

    # Step 3: ask questions
    python main.py --ask "What are the key findings?"

    # Show what child snippets matched and what parent context was sent to Claude
    python main.py --ask "Explain the methodology" --show-sources

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


def query(question: str, show_sources: bool, top_k: int) -> None:
    print(f"\nSearching for: {question!r}\n")
    result = ask_with_sources(question, top_k=top_k)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    if show_sources:
        print("\n" + "=" * 60)
        print("HOW IT WORKED  (hierarchical retrieval)")
        print("=" * 60)
        for i, src in enumerate(result["sources"], 1):
            print(
                f"\n[{i}] {src['source']}  |  page {src['page']}  |  "
                f"child match score: {src['score']}"
            )
            print(f"\n  CHILD snippet that matched your query:")
            print(f"    \"{src['matched_child'][:200]}{'...' if len(src['matched_child']) > 200 else ''}\"")
            print(f"\n  PARENT chunk sent to Claude (full context):")
            print(f"    \"{src['text'][:300]}{'...' if len(src['text']) > 300 else ''}\"")


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
        description="Hierarchical RAG — query your PDFs with Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ingest", action="store_true", help="Load PDFs and store hierarchical chunks")
    parser.add_argument("--ask", metavar="QUESTION", help="Ask a question about your documents")
    parser.add_argument("--show-sources", action="store_true", help="Show matched child + parent chunks")
    parser.add_argument("--top-k", type=int, default=5, help="Number of parent chunks to retrieve (default: 5)")
    parser.add_argument("--folder", default="rag/data/pdfs", help="Folder containing PDFs (default: rag/data/pdfs)")
    parser.add_argument("--status", action="store_true", help="Show storage stats")
    parser.add_argument("--clear", action="store_true", help="Wipe all stored chunks (use before re-ingesting)")

    args = parser.parse_args()

    if args.clear:
        clear_all()
    elif args.ingest:
        ingest(args.folder)
    elif args.ask:
        query(args.ask, show_sources=args.show_sources, top_k=args.top_k)
    elif args.status:
        status()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
