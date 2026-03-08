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
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
DB_PATH            = "rag/db"
PARENTS_COLLECTION = "parents"
CHILDREN_COLLECTION = "children"


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def _parents_col(client: chromadb.ClientAPI) -> chromadb.Collection:
    # No embedding function — we store & retrieve by ID, not by vector
    return client.get_or_create_collection(name=PARENTS_COLLECTION)


def _children_col(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=CHILDREN_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


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
        metadatas=[{"source": p["source"], "page": str(p["page"])} for p in parents],
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
    col.upsert(
        ids=[c["id"] for c in children],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source":    c["source"],
                "page":      str(c["page"]),
                "parent_id": c["parent_id"],
            }
            for c in children
        ],
    )
    print(f"Stored {len(children)} child chunks.")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Hierarchical retrieval:
      1. Find the best child chunks for the query.
      2. Fetch their parent chunks (rich context).
      3. Return de-duplicated parents, ordered by best child match score.

    Each returned dict has: text, source, page, score, matched_child (preview).
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query]).tolist()

    client = _client()
    children_col = _children_col(client)

    # Fetch more children than top_k because multiple children can share a parent
    raw = children_col.query(
        query_embeddings=query_embedding,
        n_results=min(top_k * 3, children_col.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    # Build ordered list of (parent_id, child_text, score)
    seen_parents: dict[str, dict] = {}  # parent_id -> best hit so far
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
            }

    # Keep only the top_k unique parents, ordered by score
    top_parents = sorted(seen_parents.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # Fetch full parent texts by ID
    parents_col = _parents_col(client)
    parent_ids = [h["parent_id"] for h in top_parents]
    fetched = parents_col.get(ids=parent_ids, include=["documents", "metadatas"])

    id_to_doc = dict(zip(fetched["ids"], fetched["documents"]))
    id_to_meta = dict(zip(fetched["ids"], fetched["metadatas"]))

    results = []
    for hit in top_parents:
        pid = hit["parent_id"]
        results.append({
            "text":          id_to_doc.get(pid, "[parent not found]"),
            "source":        id_to_meta.get(pid, {}).get("source", hit["source"]),
            "page":          id_to_meta.get(pid, {}).get("page", hit["page"]),
            "score":         hit["score"],
            "matched_child": hit["matched_child"],  # the small chunk that triggered this
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
