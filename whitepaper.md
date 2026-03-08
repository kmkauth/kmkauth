# Precision Retrieval for Technical Literature: Hierarchical RAG with Structured Metadata Extraction

**Draft — [Author Name], [Affiliation]**
**[Month] 2026**

---

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as the dominant pattern for grounding large language model (LLM) responses in domain-specific document corpora. However, naive implementations — fixed-size chunking applied uniformly across pages — perform poorly on technical and scientific literature, where meaning is structured hierarchically: documents contain sections, sections contain arguments, arguments span multiple paragraphs. Chunking by page boundary fractures this structure arbitrarily, degrading both retrieval precision and the quality of context delivered to the LLM.

This paper presents a retrieval architecture that addresses these failures through three complementary techniques: (1) **hierarchical parent-child chunking**, which separates the granularity of search from the granularity of context; (2) **GROBID-powered section parsing**, which replaces page-level splits with semantically coherent, labelled sections extracted via machine learning; and (3) **metadata-filtered retrieval**, which exploits structured bibliographic data — author, year, section title — to constrain search to relevant subsets of a corpus before semantic similarity is computed.

We describe the design of each component, their interactions, and the practical tradeoffs involved. The full implementation is open-source and runs entirely on local hardware with no data leaving the machine.

---

## 1. Introduction

The promise of RAG is simple: give an LLM access to a private corpus of documents and let it answer questions grounded in those documents rather than relying solely on parametric knowledge baked in during training. In practice, the gap between the promise and the reality is almost entirely determined by retrieval quality. A perfectly capable LLM fed irrelevant or truncated context will produce poor answers. Retrieval is the bottleneck.

The dominant approach in production RAG systems is to split source documents into fixed-size chunks — typically 256 to 1024 tokens — embed each chunk with a dense vector model, store the embeddings in a vector database, and at query time retrieve the top-K chunks by cosine similarity. This works adequately for short, uniform documents like FAQ pages or product descriptions. It works poorly for the kind of documents that matter most in technical and scientific fields.

Technical literature has structure. A paper on a neural architecture has an abstract, an introduction motivating the problem, a related work section, a methods section describing the architecture, an experiments section validating it, and a conclusion synthesising the results. Each section serves a different epistemic function. When a practitioner asks *"what loss function did they use?"*, the answer lives in the methods section. When they ask *"how does this compare to prior work?"*, the answer lives in the related work section. A retrieval system that treats the document as a flat bag of 512-token windows has thrown this structure away.

Two further problems compound the issue. First, **context poverty**: a retrieved chunk that perfectly matches a query may be too narrow to be useful as context for generation. The matching sentence might rely on definitions, notation, or arguments established in the surrounding paragraphs, none of which are included. The LLM receives a fragment and must guess at the missing scaffolding. Second, **corpus-level noise**: in a large corpus covering many authors and years, a query about a specific sub-topic will surface chunks from across the corpus, including papers that are merely tangentially related. Without a mechanism to exploit the structured metadata that academic papers carry — author, year, venue, section — every query searches the entire corpus indiscriminately.

This paper describes a system that addresses all three problems. The contributions are:

- A **two-level chunking scheme** (parent-child) that separates search granularity from context granularity, enabling precise retrieval without sacrificing the rich context the LLM needs.
- Integration of **GROBID**, a state-of-the-art machine learning system for structured extraction from scientific PDFs, which provides section-level segmentation and bibliographic metadata automatically.
- A **metadata-aware retrieval layer** that enables filtering by author, year, or section before or after semantic search, dramatically improving precision on focused queries.
- A **complete open-source implementation** that runs entirely locally, requiring no cloud services beyond the LLM API.

---

## 2. Background

### 2.1 Standard RAG and Its Failure Modes

The canonical RAG pipeline, introduced in Lewis et al. (2020), combines a dense retriever with a sequence-to-sequence generator. The retriever encodes both documents and queries into a shared embedding space and retrieves documents by maximum inner product search. The generator conditions on the retrieved documents to produce an answer.

In practice, most RAG deployments simplify this further: documents are chunked, each chunk is embedded independently, and retrieval is k-nearest-neighbour search in the embedding space. This approach has three well-documented failure modes in the context of long-form technical documents.

**Boundary fragmentation.** Fixed-size chunking splits at character or token counts with no regard for semantic boundaries. A paragraph spanning a chunk boundary is split mid-argument. The resulting chunks are semantically incomplete and score poorly against queries that match the whole argument rather than either half.

**Context poverty at generation time.** A 300-token chunk that precisely matches a query may depend on context from the surrounding 2000 tokens that is not retrieved. The LLM must answer with an impoverished fragment.

**Recall-precision tension.** Increasing chunk size improves context richness but degrades retrieval precision: larger chunks are less specific and match more queries spuriously. The retriever must find chunks that are simultaneously specific enough to match the query and wide enough to provide context. These objectives are in conflict.

### 2.2 Hierarchical and Multi-Granularity Chunking

Several approaches have been proposed to break the recall-precision tension. Parent-document retrieval (Langchain, 2023) stores large parent documents but indexes small child summaries, retrieving parents based on child matches. Similar ideas appear in multi-vector retrieval (Khattab et al., 2021), where documents are indexed at multiple granularities simultaneously.

The common insight is that **the unit of search need not be the unit of context**. A system can search at fine granularity — where precision is high — and expand to coarse granularity at context-assembly time — where richness is high. This paper implements a clean two-level version of this idea tuned for technical literature.

### 2.3 GROBID and Structured PDF Parsing

GROBID (Generation Of BIbliographic Data) is a machine learning library for extracting, parsing and re-structuring raw scientific documents into structured XML/TEI-encoded documents (Lopez, 2009). It has been trained on hundreds of thousands of scientific publications and achieves state-of-the-art performance on:

- Header extraction: title, authors, affiliations, abstract, keywords, publication date
- Body segmentation: section titles, paragraphs, figures, tables, equations
- Reference parsing: structured bibliographic references

GROBID exposes its functionality through a REST API. The `processFulltextDocument` endpoint accepts a PDF and returns a TEI XML document encoding the full structured content. This structured output replaces the unstructured page-by-page text extraction that pypdf provides, giving the RAG system semantic boundaries to chunk at rather than arbitrary page boundaries.

### 2.4 Vector Databases and Metadata Filtering

ChromaDB is an open-source embedding database optimised for similarity search over dense vectors. It supports persistent storage, cosine similarity via HNSW indexing, and metadata filtering via a `where` clause that operates over stored metadata fields before or alongside vector search. This allows queries of the form *"find the most semantically similar chunk to query Q, among chunks where year = 2021"*, combining semantic and structured retrieval in a single operation.

---

## 3. System Architecture

### 3.1 Ingestion Pipeline Overview

The ingestion pipeline transforms a folder of PDF files into a searchable vector store. Figure 1 shows the high-level flow.

```
PDF files (local folder)
        │
        ▼
  GROBID REST API
  (processFulltextDocument)
        │  TEI XML: title, authors, year, sections[]
        │
        │  [fallback: pypdf page extraction]
        │
        ▼
  Section/Page dicts
  {text, source, page, section, title, authors, year}
        │
        ▼
  Hierarchical Chunker
  ├── Parent chunks (~1500 chars, overlapping)
  └── Child chunks  (~300 chars, overlapping, parent_id link)
        │
        ▼
  SentenceTransformer embeddings
  (children only)
        │
        ▼
  ChromaDB (local persistent store)
  ├── "parents"  collection: stored by ID, no embeddings
  └── "children" collection: stored with embeddings + metadata
```

Every stage after GROBID is local. No document content leaves the machine.

### 3.2 GROBID Section Extraction

GROBID is called once per PDF via HTTP POST to `processFulltextDocument`. The response is a TEI XML document with a well-defined schema. The parser extracts:

**From the `<teiHeader>`:**
- `<title level="a">` — the article title
- `<author><persName>` — each author's forename and surname, normalised to "Surname, I." format
- `<date type="published" when="YYYY-...">` — publication year extracted from the `when` attribute

**From the `<body>`:**
- Each `<div>` element is one section
- `<head>` within a div gives the section title (e.g. "3.2 Loss Function")
- `<p>` elements within a div give the paragraph text
- Nested `<div>` elements (subsections) are treated as independent sections

This produces a list of section dicts: `{title: str, text: str}`. Sections with no extractable text (e.g. figure-only sections) are discarded.

**Graceful fallback.** If GROBID is unavailable — the server is not running, the request times out, or parsing fails — the system logs a warning and falls back to pypdf page-by-page extraction. The rest of the pipeline is unchanged. The only difference is that `section`, `title`, `authors`, and `year` fields are empty strings in the fallback case. This means the system degrades gracefully rather than failing hard when GROBID is not running.

### 3.3 Hierarchical Chunk Construction

Each section (or page, in fallback mode) is split into two levels of chunks using a sliding window with overlap.

**Parent chunks** are 1500 characters wide with 150 characters of overlap between consecutive parents. They are designed to provide rich, self-contained context to the LLM — wide enough to include the surrounding argument, narrow enough to avoid filling the context window with irrelevant material.

**Child chunks** are 300 characters wide with 30 characters of overlap. They are designed for precise semantic matching — narrow enough that an embedding accurately represents a single specific claim or term, rather than a mixture of several.

Every child chunk stores the ID of the parent chunk it was carved from. This is the key linkage that makes hierarchical retrieval possible.

Both levels inherit the full metadata from their source section: `source` (filename), `page` (section index or page number), `section` (section title), `title` (paper title), `authors` (comma-separated author list), `year` (publication year).

The chunk parameters are tunable constants in `loader.py` and should be adjusted based on the verbosity of the corpus. Highly mathematical papers with dense notation may benefit from larger parent chunks; narrative-heavy papers may tolerate smaller ones.

### 3.4 Embedding and Storage

Only child chunks are embedded. Parent chunks are stored by ID with no vector representation — they are never retrieved by similarity, only by explicit ID lookup. This is a deliberate design choice that avoids the cost of embedding large chunks and, more importantly, keeps the embedding space clean: every vector in the index represents a small, specific text fragment, ensuring that similarity scores are meaningful.

Embeddings are produced by `all-MiniLM-L6-v2` from the SentenceTransformers library, a 384-dimensional model optimised for semantic similarity. It runs entirely locally with no internet connection required after the first download.

ChromaDB stores the data in two collections:

- **`children`** — vectors, document text, and metadata (including `parent_id`). Queried by cosine similarity.
- **`parents`** — document text and metadata only. Retrieved by ID after child search.

ChromaDB persists both collections to disk at `rag/db/`. Incremental upserts are supported, so new documents can be added to an existing store without re-ingesting the entire corpus.

### 3.5 Metadata-Filtered Retrieval

ChromaDB's `where` clause enables structured filtering at query time. Year filtering is implemented as a pre-filter applied before vector search:

```python
where = {"year": {"$eq": "2021"}}
children_col.query(query_embeddings=..., where=where)
```

This restricts the search space to chunks from the specified year before any similarity computation, which both improves precision and reduces query latency on large corpora.

Author filtering is implemented as a post-filter in Python. ChromaDB does not natively support substring matching on metadata strings, so after retrieving the top candidates from the vector search, results are filtered to those whose `authors` field contains the query string (case-insensitive). This is less efficient than pre-filtering but handles the common case of partial name matches ("Smith" matching "Smith, J." or "Goldsmith, A.").

Both filters are optional and can be combined:

```bash
python main.py --ask "describe the loss function" \
               --filter-year 2021 \
               --filter-author "Smith"
```

---

## 4. Retrieval and Generation

### 4.1 Query-Time Flow

At query time, the system executes the following steps:

1. **Embed the query** using the same `all-MiniLM-L6-v2` model used during ingestion.
2. **Search the children collection** for the top `K × 3` most similar child chunks (fetching more than needed because multiple children can share a parent).
3. **Apply filters** — year pre-filter via ChromaDB `where`, author post-filter in Python.
4. **De-duplicate by parent ID**, keeping only the highest-scoring child for each parent.
5. **Select top-K parents** by their best child match score.
6. **Fetch parent texts** from the parents collection by ID.
7. **Assemble context** from parent texts with citation headers.
8. **Call the LLM** with the assembled context and user question.

### 4.2 De-duplication and Parent Expansion

The de-duplication step is critical. Without it, a query about a specific concept might retrieve five child chunks from the same section of the same paper, effectively sending the LLM five near-identical context passages. De-duplication ensures that the top-K results represent K distinct sections of the corpus, maximising information diversity in the context window.

After de-duplication, the system fetches the **full parent chunk** for each surviving result. This is the expansion step: the precise child that matched the query is discarded from the context, and the wider parent that contains it is used instead. The LLM never sees the narrow child chunk — it sees the rich surrounding context.

The matched child snippet is retained separately for display in `--show-sources` mode, giving the user transparency into why each source was retrieved.

### 4.3 Citation-Aware Prompting

The context passed to the LLM is formatted to include bibliographic metadata on every source:

```
[Source 1: Smith, J., Jones, A. (2021) — "Attention Is All You Need"  §Experiments
 | file: attention.pdf | relevance 0.8923]

[section text...]

---

[Source 2: ...]
```

The system prompt instructs the LLM to cite author names and years when referring to sources. This produces answers with inline attribution rather than vague references to "the document", which is particularly important when the corpus contains multiple papers on related topics that may disagree.

---

## 5. Evaluation

> **[TODO for author]**
>
> This section requires empirical results from your specific corpus. Suggested evaluation protocol:
>
> **5.1 Dataset**
> Describe your PDF corpus: number of papers, domain, year range, average length.
>
> **5.2 Baseline**
> Flat RAG: pypdf page splits, 512-char chunks, same embedding model, no metadata. Same LLM.
>
> **5.3 Metrics**
> - *Answer relevance*: human evaluation or LLM-as-judge (1–5 scale) on 20–50 representative questions
> - *Source precision*: for questions with known ground-truth sources, what fraction of retrieved parents contain the answer?
> - *Citation accuracy*: does the answer correctly attribute claims to the right author/year?
> - *Retrieval latency*: wall-clock time per query
>
> **5.4 Suggested ablations**
> - Full system vs. no GROBID (pypdf fallback)
> - Full system vs. flat chunking (no parent expansion)
> - With vs. without metadata filters on filtered queries
>
> Strong expected results: GROBID section parsing should show the largest gain on questions
> that span section boundaries; parent expansion should show the largest gain on questions
> requiring multi-sentence reasoning.

---

## 6. Discussion

### 6.1 When GROBID Section Parsing Helps Most

The benefit of GROBID section parsing is largest when the target question aligns with the document's own section structure. Questions like *"what datasets were used?"* or *"what is the model architecture?"* map directly onto standard paper sections (Experiments, Methods). In these cases, section-level chunking ensures the relevant text is never split across chunk boundaries, and the section title provides a strong semantic signal that boosts retrieval precision.

The benefit is smallest for questions that cut across sections — for example, *"what are the limitations of this approach?"* may be partially answered in the introduction, partially in the experiments, and partially in a dedicated limitations section if one exists. Hierarchical retrieval mitigates this by retrieving multiple parents, but cross-section synthesis remains a fundamentally harder retrieval problem.

### 6.2 Tradeoffs in Chunk Size Tuning

The 1500/300 character split used in this implementation is a reasonable default for typical NLP and ML papers, which tend to have moderately dense prose. Corpora with different characteristics may benefit from different settings:

- **Dense mathematical text** (e.g. signal processing, control theory): larger parent chunks (2000–3000 chars) to include more context around equations; smaller child chunks may be counterproductive if individual sentences are too terse to embed meaningfully.
- **Verbose narrative text** (e.g. survey papers, review articles): standard settings work well; child chunks of 300 chars capture individual claims cleanly.
- **Highly structured text** (e.g. standards documents, technical specifications): consider section-as-parent (no further splitting of GROBID sections into parent chunks) and sentence-level children.

### 6.3 Limitations

**Scanned PDFs.** GROBID and pypdf both require machine-readable PDFs. Scanned documents require OCR pre-processing (e.g. Tesseract, AWS Textract) before ingestion.

**Non-standard layouts.** Two-column layouts, papers with extensive figure/table content, and documents with unusual section structures may cause GROBID to mis-segment sections or lose text. The pypdf fallback provides a safety net but without structural benefits.

**Author disambiguation.** Author filtering is currently a simple substring match. "Smith" will match "Goldsmith". A production system should use ORCID identifiers or co-author graph disambiguation.

**GROBID dependency.** GROBID requires Java and a running server process (or Docker). For deployments where this is impractical, the pypdf fallback is the only option, and the metadata-filtering benefits are unavailable.

### 6.4 Extensions

**RPCA + PCA embedding compression.** Before storing child embeddings, Robust PCA can be used to strip noise and outliers from the embedding matrix, followed by PCA dimensionality reduction (e.g. 384 → 128 dimensions). This reduces storage and query latency with minimal loss of retrieval quality, and is particularly valuable for large corpora.

**Cross-document reasoning.** The current system treats each retrieved parent independently. A natural extension is to include a synthesis step: after retrieving the top-K parents, prompt the LLM to identify agreements and contradictions across sources before answering. This is valuable for literature review queries.

**Re-ranking.** A cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) can be applied to the top-K candidates after initial retrieval to re-score them based on the full query–document pair. This adds latency but substantially improves precision on complex queries.

**Hybrid search.** Combining dense vector search with sparse BM25 retrieval captures both semantic similarity and exact keyword matches. This is particularly useful for queries involving specific technical terms, model names, or dataset identifiers that may not embed well.

---

## 7. Conclusion

Flat RAG fails technical literature because it ignores the structure that makes technical documents useful. Page boundaries are not semantic boundaries. A paper is not a sequence of pages; it is a hierarchy of arguments, and retrieving fragments of arguments without their context produces impoverished responses.

The architecture described in this paper addresses this directly. GROBID provides semantically coherent, labelled sections as the unit of chunking. Hierarchical parent-child indexing separates the precision of search from the richness of context. Metadata-filtered retrieval exploits the structured bibliographic information that GROBID extracts, enabling corpus-level navigation by author, year, and section that pure semantic search cannot provide.

The full implementation runs locally, requires no cloud services for storage or retrieval, and degrades gracefully when GROBID is unavailable. It is intended as a practical foundation for organisations that need to query large private corpora of scientific and technical documents without sending that material to external services.

---

## References

> **[TODO for author]** — populate with your actual citations.
> Suggested entries to include:

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Lopez, P. (2009). *GROBID: Combining Automatic Bibliographic Data Recognition and Term Extraction for Scholarship Publications.* ECDL 2009.
- Khattab, O., & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR 2020.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
- Trung, L. (2023). *LangChain Parent Document Retriever.* LangChain documentation.
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data.

---

## Appendix A: Running the System

```bash
# 1. Start GROBID (Docker)
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Drop PDFs into the data folder
cp /your/papers/*.pdf rag/data/pdfs/

# 4. Ingest
python main.py --ingest

# 5. Query
python main.py --ask "What loss function is used?" --show-sources

# Filter by year or author
python main.py --ask "describe the architecture" --filter-year 2021
python main.py --ask "regularisation approach"   --filter-author "Smith"

# Re-ingest after adding new PDFs
python main.py --clear && python main.py --ingest
```

## Appendix B: Configuration Reference

| Parameter | Location | Default | Description |
|---|---|---|---|
| `PARENT_CHUNK_SIZE` | `loader.py` | 1500 | Characters per parent chunk |
| `PARENT_OVERLAP` | `loader.py` | 150 | Overlap between parent chunks |
| `CHILD_CHUNK_SIZE` | `loader.py` | 300 | Characters per child chunk |
| `CHILD_OVERLAP` | `loader.py` | 30 | Overlap between child chunks |
| `EMBEDDING_MODEL` | `vectorstore.py` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `GROBID_URL` | `grobid.py` | `http://localhost:8070` | GROBID server endpoint |
| `DB_PATH` | `vectorstore.py` | `rag/db` | ChromaDB storage location |
| `--top-k` | CLI | 5 | Parent chunks retrieved per query |
