"""
main.py — Command-line interface for the RAG framework.

Usage:
    # Step 1: Ingest your PDFs (drop them in rag/data/pdfs/ first)
    python main.py --ingest

    # Step 2: Ask a question
    python main.py --ask "What is the main topic of the document?"

    # Ask and show the source chunks used
    python main.py --ask "Summarize the key findings" --show-sources

    # Check how many chunks are stored
    python main.py --status
"""

import argparse
import sys
from rag.loader import load_pdfs_from_folder
from rag.vectorstore import add_chunks, collection_size
from rag.query import ask_with_sources


def ingest(folder: str) -> None:
    print(f"\nIngesting PDFs from '{folder}'...")
    chunks = load_pdfs_from_folder(folder)
    if chunks:
        add_chunks(chunks)
        print(f"\nDone! {len(chunks)} chunks are now searchable.")
    else:
        print("Nothing ingested.")


def query(question: str, show_sources: bool, top_k: int) -> None:
    print(f"\nSearching for: {question!r}\n")
    result = ask_with_sources(question, top_k=top_k)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    if show_sources:
        print("\n" + "=" * 60)
        print("SOURCES USED")
        print("=" * 60)
        for i, src in enumerate(result["sources"], 1):
            print(f"\n[{i}] {src['source']}  |  page {src['page']}  |  relevance {src['score']}")
            print(f"    {src['text'][:200]}{'...' if len(src['text']) > 200 else ''}")


def status() -> None:
    count = collection_size()
    print(f"\nVector store contains {count} chunks.")
    if count == 0:
        print("Tip: run  python main.py --ingest  after dropping PDFs into rag/data/pdfs/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG framework — query your PDFs with Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ingest", action="store_true", help="Load PDFs and store embeddings")
    parser.add_argument("--ask", metavar="QUESTION", help="Ask a question about your documents")
    parser.add_argument("--show-sources", action="store_true", help="Print the source chunks alongside the answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--folder", default="rag/data/pdfs", help="Folder containing PDFs (default: rag/data/pdfs)")
    parser.add_argument("--status", action="store_true", help="Show how many chunks are stored")

    args = parser.parse_args()

    if args.ingest:
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
