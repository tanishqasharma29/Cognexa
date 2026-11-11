# Week 2 — RAG Retrieval and Quiz Generation Prototype

## 1. Objective

The objective of Week 2 was to build a minimal RAG-based quiz generation prototype that converts source documents into traceable evidence and uses that evidence to generate grounded quiz questions.

The implemented pipeline is:

Documents
→ Text Extraction
→ Chunking
→ Embeddings + BM25
→ Dense + Lexical Retrieval
→ Hybrid RRF
→ Cross-Encoder Reranking
→ Context Selection
→ Gemini Quiz Generation
→ Source-Cited Questions

---

## 2. Source Documents

The prototype currently uses four source PDFs:

- d1.pdf — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- d2.pdf — Attention Is All You Need
- d3.pdf — MIT 6.006 Lecture 4: Hashing
- d4.pdf — Scanned Chemical Reaction Network material

The first three documents provide selectable text. d4.pdf is scanned/image-based and is reserved for the OCR/data-quality component.

---

## 3. Checked Extraction

PDF pages were extracted individually using PyMuPDF.

Each extracted page stores:

- document_id
- source_file
- page
- text
- character_count
- word_count

Output:

data/processed/extracted_pages.json

---

## 4. Chunking

A fixed-size chunking strategy was implemented.

Configuration:

- Chunk size: 700 words
- Chunk overlap: 100 words

Each chunk stores:

- chunk_id
- document_id
- source_file
- page
- heading
- text
- word_count

Output:

data/processed/chunks.json

---

## 5. Dense Retrieval

Dense retrieval uses:

Model:
all-MiniLM-L6-v2

Similarity:
Normalized embedding dot product

The system retrieves the top 5 candidate chunks for each query.

---

## 6. Lexical Retrieval

Lexical retrieval uses BM25.

The same corpus and queries are used so that lexical and dense retrieval can be compared fairly.

---

## 7. Hybrid Retrieval

Dense and lexical results are combined using Reciprocal Rank Fusion (RRF).

RRF parameter:

k = 60

The hybrid method preserves the component ranks and scores.

---

## 8. Reranking

The hybrid shortlist is reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

The reranker evaluates query-passage relevance and produces the final ranking.

---

## 9. Context Selection

The final context selection step:

- removes duplicate chunks
- keeps the strongest passages
- limits the context to approximately 1800 words
- preserves source/page/chunk metadata

Output:

results/selected_context.json

---

## 10. Quiz Generation

Gemini 3.8 Flash is used for grounded quiz generation.

The generation prompt requires:

- question
- answer
- supporting_excerpt
- source_file
- page
- chunk_id
- difficulty

The model is explicitly instructed to use only retrieved evidence and not invent unsupported facts.

Output:

results/generated_quiz.json

---

## 11. End-to-End Example

Query:

What is Retrieval-Augmented Generation?

The system retrieved evidence from d1.pdf and generated questions such as:

"What two types of memory are combined in retrieval-augmented generation (RAG)?"

Answer:

"RAG combines pre-trained parametric memory and non-parametric memory."

The generated question contains a supporting excerpt and the exact source document, page, and chunk ID.

---

## 12. Retrieval Benchmark

Ten benchmark queries were evaluated:

- 9 answerable queries
- 1 unsupported query

Methods compared:

1. Lexical / BM25
2. Dense retrieval
3. Hybrid RRF
4. Hybrid + reranking

### Results

| Method | Hit@5 | MRR | Avg Time (s) |
|---|---:|---:|---:|
| Lexical | 1.000 | 0.531 | 0.0001 |
| Dense | 1.000 | 0.500 | 0.1302 |
| Hybrid | 1.000 | 0.611 | ~0.0000 |
| Hybrid + Reranking | 1.000 | 0.648 | 0.0680 |

### Interpretation

All methods achieved Hit@5 = 1.000 on the answerable benchmark queries.

MRR showed a clearer difference in ranking quality. Hybrid retrieval improved MRR compared with lexical or dense retrieval alone, and hybrid retrieval followed by reranking produced the highest MRR of 0.648.

Therefore, within this small benchmark, hybrid retrieval plus reranking produced the strongest ranking performance.

---

## 13. Unsupported Query Test

Unsupported query:

"What is the capital of France?"

All retrieval methods returned candidate passages because retrieval systems always rank available corpus passages even when the corpus does not contain the answer.

However, the grounded quiz-generation layer correctly refused to generate a question:

{
  "questions": [],
  "insufficient_evidence": true
}

This demonstrates that retrieval and generation should be treated separately: retrieval may return weak candidates, while the generation layer can still abstain when the evidence is insufficient.

---

## 14. Twenty-Question Quality Evaluation

A 20-question quiz-generation run was evaluated.

Results:

| Criterion | Result |
|---|---:|
| Complete fields | 20/20 |
| Evidence present | 20/20 |
| Citation present | 20/20 |
| Answerable | 20/20 |
| Valid difficulty | 20/20 |
| Potential duplicates | 0 |

The generated questions contained the required structural fields and traceable source information.

---

## 15. Observed Limitation

The 20-question request covered:

- RAG
- Transformers
- Hashing

However, the generated questions were concentrated on RAG and Transformers, with no hashing questions.

This indicates a limitation in multi-topic context selection: when the retrieved context is dominated by a subset of highly relevant passages, the final quiz may underrepresent other requested topics.

This should be addressed in a later iteration using topic-aware retrieval or per-topic context allocation.

---

## 16. Current Prototype Structure

prototype/

├── data/
│   ├── raw/
│   └── processed/
│
├── indexes/
│   └── dense_index.npz
│
├── results/
│   ├── selected_context.json
│   ├── generated_quiz.json
│   ├── benchmark_queries.json
│   ├── benchmark_results.json
│   ├── quiz_evaluation.json
│   └── week2_report.md
│
├── src/
│   ├── extract.py
│   ├── chunk.py
│   ├── dense_retrieval.py
│   ├── lexical_retrieval.py
│   ├── hybrid_retrieval.py
│   ├── reranking.py
│   ├── context_selection.py
│   ├── quiz_generator.py
│   ├── benchmark.py
│   ├── benchmark_summary.py
│   └── evaluate_quiz.py
│
└── .venv/

---

## 17. Conclusion

The Week 2 prototype successfully demonstrates an end-to-end evidence-grounded quiz generation pipeline.

The strongest retrieval configuration in the current benchmark was hybrid retrieval followed by cross-encoder reranking, achieving an MRR of 0.648.

The quiz generation layer preserved source traceability and successfully abstained from generating an answer when the evidence was insufficient.

The primary current limitation is balanced coverage of multi-topic requests, which will require topic-aware retrieval or context allocation in a future iteration.

### Multi-Topic Retrieval Improvement

The initial global context-selection strategy produced 20 questions with
RAG and Transformer coverage but no Hashing questions.

A topic-aware context-selection strategy was therefore introduced. Retrieval
and reranking were performed separately for RAG, Transformers, and Hashing.
At least one high-ranked chunk from each requested topic was first guaranteed,
after which remaining context slots were filled according to reranking score.

For the final 20-question generation:

- RAG: 7 questions
- Transformers: 5 questions
- Hashing: 8 questions
- Total: 20 questions

The generated questions remained fully traceable to the source documents.

### Final Quiz Quality Evaluation

The improved 20-question quiz was evaluated using the automated quality
evaluation script.

| Metric | Result |
|---|---:|
| Total questions | 20 |
| Complete fields | 20/20 |
| Evidence present | 20/20 |
| Citation present | 20/20 |
| Answerable | 20/20 |
| Valid difficulty | 20/20 |
| Potential duplicates | 0 |

The evaluation indicates that all generated questions contained the required
metadata, supporting evidence, valid source citations, answerable content,
and valid difficulty labels, with no potential duplicates detected.