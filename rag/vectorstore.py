"""
vectorstore.py — Two-collection ChromaDB store for hierarchical RAG.

Collections:
  "parents"  — large chunks stored by ID, NOT searched by vector.
               We look these up by ID after finding relevant children.

  "children" — small chunks WITH vector embeddings.
               These are what we actually search against the query.

Retrieval flow:
  1. Embed the user's query
  2. Search "children" collection → get top-K child chunks
  3. De-duplicate their parent_ids
  4. Fetch matching parent chunks from "parents" by ID
  5. Send parent text (rich context) to Claude

Metadata stored per chunk: source, page, section, title, authors, year
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
DB_PATH             = "rag/db"
PARENTS_COLLECTION  = "parents"
CHILDREN_COLLECTION = "children"

# Metadata keys that flow from chunks into ChromaDB (all stored as strings)
_META_KEYS = ("source", "page", "section", "title", "authors", "year")


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def _parents_col(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(name=PARENTS_COLLECTION)


def _children_col(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=CHILDREN_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _chroma_meta(chunk: dict) -> dict:
    """Return a ChromaDB-safe metadata dict (all values must be str/int/float)."""
    return {k: str(chunk.get(k, "") or "") for k in _META_KEYS}


# ── Ingestion ─────────────────────────────────────────────────────────────────

def add_parents(parents: list[dict]) -> None:
    """Store parent chunks by ID (no embeddings needed)."""
    if not parents:
        return
    client = _client()
    col = _parents_col(client)
    col.upsert(
        ids=[p["id"] for p in parents],
        documents=[p["text"] for p in parents],
        metadatas=[_chroma_meta(p) for p in parents],
    )
    print(f"Stored {len(parents)} parent chunks.")


def add_children(children: list[dict]) -> None:
    """Embed child chunks and store them with a reference to their parent."""
    if not children:
        return
    print(f"Embedding {len(children)} child chunks with '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in children]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = _client()
    col = _children_col(client)

    # Build metadata: include parent_id on top of the standard meta keys
    metadatas = []
    for c in children:
        meta = _chroma_meta(c)
        meta["parent_id"] = c["parent_id"]
        metadatas.append(meta)

    col.upsert(
        ids=[c["id"] for c in children],
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Stored {len(children)} child chunks.")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = 5,
    filter_year: str | None = None,
    filter_author: str | None = None,
) -> list[dict]:
    """
    Hierarchical retrieval:
      1. Find the best child chunks for the query (optional year filter via ChromaDB).
      2. Fetch their parent chunks (rich context).
      3. Optionally post-filter by author substring.
      4. Return de-duplicated parents, ordered by best child match score.

    Each returned dict: text, source, page, section, title, authors, year,
                        score, matched_child.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query]).tolist()

    client = _client()
    children_col = _children_col(client)

    where = None
    if filter_year:
        where = {"year": {"$eq": filter_year}}

    raw = children_col.query(
        query_embeddings=query_embedding,
        n_results=min(top_k * 3, children_col.count() or 1),
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    # Build ordered map: parent_id → best hit
    seen_parents: dict[str, dict] = {}
    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        parent_id = meta["parent_id"]
        score = round(1 - dist, 4)
        if parent_id not in seen_parents or score > seen_parents[parent_id]["score"]:
            seen_parents[parent_id] = {
                "parent_id":     parent_id,
                "matched_child": doc,
                "score":         score,
                "source":        meta.get("source", ""),
                "page":          meta.get("page", "?"),
                "section":       meta.get("section", ""),
                "title":         meta.get("title", ""),
                "authors":       meta.get("authors", ""),
                "year":          meta.get("year", ""),
            }

    top_parents = sorted(seen_parents.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # Optional author post-filter (substring match)
    if filter_author:
        needle = filter_author.lower()
        top_parents = [h for h in top_parents if needle in h.get("authors", "").lower()]

    # Fetch full parent texts
    parents_col = _parents_col(client)
    parent_ids = [h["parent_id"] for h in top_parents]
    if not parent_ids:
        return []
    fetched = parents_col.get(ids=parent_ids, include=["documents", "metadatas"])

    id_to_doc  = dict(zip(fetched["ids"], fetched["documents"]))
    id_to_meta = dict(zip(fetched["ids"], fetched["metadatas"]))

    results = []
    for hit in top_parents:
        pid  = hit["parent_id"]
        meta = id_to_meta.get(pid, {})
        results.append({
            "text":          id_to_doc.get(pid, "[parent not found]"),
            "source":        meta.get("source", hit["source"]),
            "page":          meta.get("page",   hit["page"]),
            "section":       meta.get("section", hit["section"]),
            "title":         meta.get("title",   hit["title"]),
            "authors":       meta.get("authors", hit["authors"]),
            "year":          meta.get("year",    hit["year"]),
            "score":         hit["score"],
            "matched_child": hit["matched_child"],
        })

    return results


# ── Utilities ─────────────────────────────────────────────────────────────────

def collection_sizes() -> dict[str, int]:
    client = _client()
    return {
        "parents":  _parents_col(client).count(),
        "children": _children_col(client).count(),
    }


def clear_all() -> None:
    """Delete both collections (useful for re-ingesting from scratch)."""
    client = _client()
    for name in (PARENTS_COLLECTION, CHILDREN_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass
    print("Cleared all collections.")
