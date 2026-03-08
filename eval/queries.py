"""
eval/queries.py — Test query bank for white paper validation.

Organised into six categories that map directly to the white paper's §5 evaluation:

  1. FACTUAL        — single-source, specific fact retrieval
  2. SECTION_TARGETED — queries that map to a known document section
  3. CROSS_DOC      — synthesis across multiple papers
  4. METADATA       — author/year filter queries
  5. HALLUCINATION  — questions the corpus cannot answer (should gracefully refuse)
  6. CITATION       — validates that answers name sources correctly

HOW TO CUSTOMISE:
  - Replace placeholder questions with ones relevant to YOUR corpus.
  - Fill "expected_keywords" with terms that must appear in a correct answer.
  - Fill "expected_source_contains" with filename substrings or author names.
  - For HALLUCINATION queries, leave expected_keywords empty and set
    "expect_refusal": True — the system should say it cannot find the answer.
  - "ground_truth" is a 1-3 sentence gold answer for LLM-as-judge comparison
    (optional; leave "" if unknown before running).
"""

# ---------------------------------------------------------------------------
# Test record schema
# ---------------------------------------------------------------------------
# {
#   "id":                    str   — unique test ID
#   "category":              str   — one of the six categories
#   "question":              str   — the query sent to the RAG system
#   "filter_year":           str|None
#   "filter_author":         str|None
#   "top_k":                 int
#   "expected_keywords":     list[str]  — keywords that MUST appear in the answer
#   "forbidden_keywords":    list[str]  — keywords that must NOT appear (hallucination check)
#   "expected_source_contains": list[str]  — substrings expected in source filenames/authors
#   "expect_refusal":        bool  — True if the system should say it cannot answer
#   "ground_truth":          str   — gold reference answer (optional)
#   "notes":                 str   — human-readable intent of the test
# }

QUERIES = [

    # ------------------------------------------------------------------
    # CATEGORY 1: FACTUAL
    # Single-hop, single-source factual retrieval.
    # Replace these with questions whose answers exist verbatim in one of
    # your ingested PDFs.
    # ------------------------------------------------------------------
    {
        "id": "F-01",
        "category": "FACTUAL",
        "question": "What dataset was used to train the model described in the paper?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],          # e.g. ["ImageNet", "CIFAR-10"]
        "forbidden_keywords": [],
        "expected_source_contains": [],   # e.g. ["attention.pdf"]
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Checks that the system retrieves the correct dataset name from the Methods/Experiments section.",
    },
    {
        "id": "F-02",
        "category": "FACTUAL",
        "question": "What is the reported top-1 accuracy of the proposed model on the benchmark?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],          # e.g. ["92.3%", "93", "accuracy"]
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Numeric fact retrieval — verifies precision on exact figures.",
    },
    {
        "id": "F-03",
        "category": "FACTUAL",
        "question": "What loss function is used during training?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],          # e.g. ["cross-entropy", "contrastive"]
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Maps to the Methods §Loss Function subsection — tests section-level retrieval.",
    },
    {
        "id": "F-04",
        "category": "FACTUAL",
        "question": "How many parameters does the largest model variant have?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],          # e.g. ["175B", "7 billion"]
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Tests retrieval of a specific numeric configuration detail.",
    },
    {
        "id": "F-05",
        "category": "FACTUAL",
        "question": "What hardware was used for training?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],          # e.g. ["A100", "V100", "TPU"]
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Often buried in an implementation details subsection — tests depth of retrieval.",
    },

    # ------------------------------------------------------------------
    # CATEGORY 2: SECTION_TARGETED
    # Queries that should map to a specific named section in the paper.
    # GROBID section labels should appear in the source metadata.
    # ------------------------------------------------------------------
    {
        "id": "S-01",
        "category": "SECTION_TARGETED",
        "question": "Describe the methodology and experimental setup.",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Should retrieve from 'Methods' or 'Experimental Setup' sections specifically.",
    },
    {
        "id": "S-02",
        "category": "SECTION_TARGETED",
        "question": "What prior work does the paper build on?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Should retrieve from 'Related Work' — validates section-targeted routing.",
    },
    {
        "id": "S-03",
        "category": "SECTION_TARGETED",
        "question": "What are the stated limitations of the approach?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Limitations are often in Conclusion or a dedicated subsection.",
    },
    {
        "id": "S-04",
        "category": "SECTION_TARGETED",
        "question": "What ablation experiments were run and what did they show?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Maps to Ablations subsection of Experiments — critical for validating GROBID subsection parsing.",
    },

    # ------------------------------------------------------------------
    # CATEGORY 3: CROSS_DOC
    # Synthesis queries that require information from MORE THAN ONE paper.
    # Good answers will cite multiple sources.
    # ------------------------------------------------------------------
    {
        "id": "C-01",
        "category": "CROSS_DOC",
        "question": "How do the papers in this corpus differ in their approach to regularisation?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 7,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Should surface multiple papers — validates cross-document retrieval.",
    },
    {
        "id": "C-02",
        "category": "CROSS_DOC",
        "question": "Summarise the main contributions of all papers in the corpus.",
        "filter_year": None,
        "filter_author": None,
        "top_k": 10,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Broad synthesis — validates that multiple distinct sources are retrieved.",
    },
    {
        "id": "C-03",
        "category": "CROSS_DOC",
        "question": "Which papers report the best benchmark performance and on what tasks?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 7,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Requires comparing numeric results across papers.",
    },

    # ------------------------------------------------------------------
    # CATEGORY 4: METADATA_FILTERED
    # Tests year and author filters. Replace placeholder values with real
    # author names and years from your corpus.
    # ------------------------------------------------------------------
    {
        "id": "M-01",
        "category": "METADATA_FILTERED",
        "question": "What are the key findings of the papers published in 2022?",
        "filter_year": "2022",            # <-- change to a year in your corpus
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Validates year pre-filter — all sources should have year=2022.",
    },
    {
        "id": "M-02",
        "category": "METADATA_FILTERED",
        "question": "Describe the methodology used in the most recent papers.",
        "filter_year": "2023",            # <-- change to match your corpus
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Year filter on methodology queries.",
    },
    {
        "id": "M-03",
        "category": "METADATA_FILTERED",
        "question": "What models does this author propose?",
        "filter_year": None,
        "filter_author": "REPLACE_WITH_AUTHOR_SURNAME",  # <-- replace
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": ["REPLACE_WITH_AUTHOR_SURNAME"],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Validates author post-filter — all sources should be by this author.",
    },
    {
        "id": "M-04",
        "category": "METADATA_FILTERED",
        "question": "What contributions did this author make in 2021?",
        "filter_year": "2021",            # <-- change
        "filter_author": "REPLACE_WITH_AUTHOR_SURNAME",  # <-- replace
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Combined year+author filter — strictest metadata test.",
    },

    # ------------------------------------------------------------------
    # CATEGORY 5: HALLUCINATION
    # These questions CANNOT be answered from the corpus.
    # The system MUST say it doesn't have the information rather than
    # confabulating an answer. Tests grounding discipline.
    # ------------------------------------------------------------------
    {
        "id": "H-01",
        "category": "HALLUCINATION",
        "question": "What is the GDP of Brazil in 2023?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": ["trillion", "billion", "USD", "real", "BRL"],  # would indicate hallucination
        "expected_source_contains": [],
        "expect_refusal": True,
        "ground_truth": "",
        "notes": "Completely off-topic — system must refuse cleanly, not invent economic data.",
    },
    {
        "id": "H-02",
        "category": "HALLUCINATION",
        "question": "Who won the FIFA World Cup in 2022?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": ["Argentina", "France", "Mbappe", "Messi", "won", "final"],
        "expected_source_contains": [],
        "expect_refusal": True,
        "ground_truth": "",
        "notes": "Factual world-knowledge question with no plausible link to a technical corpus.",
    },
    {
        "id": "H-03",
        "category": "HALLUCINATION",
        "question": "What is the boiling point of tungsten?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": ["5555", "5828", "degrees", "Celsius", "Kelvin", "boiling"],
        "expected_source_contains": [],
        "expect_refusal": True,
        "ground_truth": "",
        "notes": "Specific scientific fact extremely unlikely to appear in an ML/NLP corpus.",
    },
    {
        "id": "H-04",
        "category": "HALLUCINATION",
        "question": "List every paper published by [fake author name] on quantum entanglement.",
        "filter_year": None,
        "filter_author": "Zxqvbn Abcdefgh",   # deliberately nonsense name
        "top_k": 5,
        "expected_keywords": [],
        "forbidden_keywords": [],
        "expected_source_contains": [],
        "expect_refusal": True,
        "ground_truth": "",
        "notes": "Non-existent author — filter should return no results; system should refuse gracefully.",
    },

    # ------------------------------------------------------------------
    # CATEGORY 6: CITATION
    # Validates that answers correctly attribute claims to named authors
    # and years, not just to vague 'the document' references.
    # ------------------------------------------------------------------
    {
        "id": "CI-01",
        "category": "CITATION",
        "question": "Who first proposed the attention mechanism as described in your corpus?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],          # e.g. ["Vaswani", "2017"]
        "forbidden_keywords": ["the document", "the paper", "the author"],  # vague non-citations
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Answer must name specific authors and years, not generic references.",
    },
    {
        "id": "CI-02",
        "category": "CITATION",
        "question": "Which authors introduced the concept of dropout regularisation according to your documents?",
        "filter_year": None,
        "filter_author": None,
        "top_k": 5,
        "expected_keywords": [],          # e.g. ["Srivastava", "Hinton", "2014"]
        "forbidden_keywords": ["the document", "the paper"],
        "expected_source_contains": [],
        "expect_refusal": False,
        "ground_truth": "",
        "notes": "Citation accuracy check — author names and years must appear.",
    },
]
