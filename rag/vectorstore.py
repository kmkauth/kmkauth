"""
vectorstore.py — Embed text chunks and store/retrieve them with ChromaDB.

Uses sentence-transformers for local embeddings (no API key needed for this part).
"""

import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Local embedding model — downloads once, then cached
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "rag/db"
COLLECTION_NAME = "documents"


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(chunk: dict) -> str:
    """Stable unique ID for a chunk based on its content."""
    key = f"{chunk['source']}:p{chunk['page']}:s{chunk.get('chunk_start', 0)}"
    return hashlib.md5(key.encode()).hexdigest()


def add_chunks(chunks: list[dict]) -> None:
    """
    Embed chunks and upsert them into ChromaDB.
    Skips chunks that are already stored (idempotent).
    """
    if not chunks:
        print("No chunks to add.")
        return

    print(f"Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = _get_client()
    collection = _get_collection(client)

    ids = [_chunk_id(c) for c in chunks]
    metadatas = [{"source": c["source"], "page": str(c["page"])} for c in chunks]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at '{DB_PATH}'.")


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and return the top_k most relevant chunks.

    Returns a list of dicts with keys: text, source, page, score.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query]).tolist()

    client = _get_client()
    collection = _get_collection(client)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", ""),
            "page": meta.get("page", "?"),
            "score": round(1 - dist, 4),  # cosine similarity
        })
    return hits


def collection_size() -> int:
    """Return how many chunks are currently stored."""
    client = _get_client()
    collection = _get_collection(client)
    return collection.count()
