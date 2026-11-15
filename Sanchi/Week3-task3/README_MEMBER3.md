# Member 3 — Retrieval Pipeline and Indexing

## 1. Objective

This document describes the retrieval pipeline and indexing work completed by Member 3 for the Cognexa project. The pipeline supports a controlled comparison of retrieval quality before and after adding OCR-extracted documents.

The same chunking strategy, embedding model, retrieval methods, and retrieval settings are used for Dataset A and Dataset B so that the comparison remains consistent.

## 2. Dataset configuration

Two datasets are prepared:

| Dataset | Contents | Documents | Pages | Chunks |
|---|---|---:|---:|---:|
| Dataset A | Original PDFs: d1.pdf, d2.pdf, d3.pdf, d4.pdf | 4 | 12 | 12 |
| Dataset B | Dataset A plus nine additional OCR documents | 13 | 151 | 151 |

D4 is included in both datasets. Its OCR output is stored in `data/processed/d4_ocr_pages.json`.

The dataset-generation script creates:

- `data/processed/dataset_A/pages.json`
- `data/processed/dataset_A/chunks.json`
- `data/processed/dataset_B/pages.json`
- `data/processed/dataset_B/chunks.json`

### Chunking configuration

- Chunk size: 700 words
- Overlap: 100 words
- The same chunking configuration is applied to both datasets.
- Each chunk retains document, source-file, page, and chunk identifiers for traceability.

In the current data, each page produces one chunk, so the page and chunk counts are equal.

## 3. Retrieval pipeline

The retrieval pipeline contains four methods.

### 3.1 BM25 lexical retrieval

BM25 ranks chunks using lexical term matching. It is useful when the query and relevant document contain matching words or phrases.

Implementation: `src/lexical_retrieval.py`

### 3.2 Dense retrieval

Dense retrieval converts each chunk and query into vector embeddings and ranks chunks by semantic similarity.

- Embedding model: `all-MiniLM-L6-v2`
- Embedding dimension: 384
- Embeddings are normalized.
- Separate indexes are created for Dataset A and Dataset B.

Index files:

- `indexes/dense_index_dataset_a.npz`
- `indexes/dense_index_dataset_b.npz`

Implementation: `src/dense_retrieval.py`

### 3.3 Hybrid retrieval using Reciprocal Rank Fusion (RRF)

Hybrid retrieval combines BM25 and dense retrieval. Reciprocal Rank Fusion (RRF) combines the rankings from both methods.

- Candidate count: 10
- Final displayed results: 5
- RRF constant: 60

RRF helps combine lexical matching and semantic similarity without directly comparing their differently scaled raw scores.

Implementation: `src/hybrid_retrieval.py`

### 3.4 Hybrid retrieval with reranking

This method first obtains candidates using dense and BM25 retrieval, combines them using RRF, and then reranks the candidates with a cross-encoder.

- Initial candidate count: 10
- Final results: 5
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

The cross-encoder evaluates the query and candidate text together to produce a more query-specific relevance score.

Implementation: `src/reranking.py`

## 4. Benchmark configuration

The benchmark uses the same query set and retrieval configuration for both datasets.

- Benchmark queries: `results/benchmark_queries.json`
- Number of queries: 12
- Retrieval methods: lexical, dense, hybrid, hybrid with reranking
- Candidate count saved per method: up to 10
- Final retrieval display: top 5

The benchmark scripts save document and page traceability fields with each retrieved result:

- `document_id`
- `source_file`
- `page`
- `chunk_id`
- `score`
- `rank`

This makes it possible to identify the exact source document and page for every retrieved chunk.

## 5. Main scripts

| Script | Purpose |
|---|---|
| `src/ocr_d4.py` | Runs OCR for D4 and saves its page text |
| `src/week3_datasets.py` | Builds and validates Dataset A and Dataset B |
| `src/dense_retrieval.py` | Runs dense retrieval and creates separate dense indexes |
| `src/lexical_retrieval.py` | Runs BM25 lexical retrieval |
| `src/hybrid_retrieval.py` | Runs BM25 + dense retrieval with RRF |
| `src/reranking.py` | Runs hybrid retrieval followed by cross-encoder reranking |
| `src/benchmark.py` | Benchmarks all four methods on Dataset A |
| `src/benchmark_dataset_b.py` | Benchmarks all four methods on Dataset B |

## 6. Generated results and indexes

### Benchmark outputs

- Dataset A: `results/benchmark_results.json`
- Dataset B: `results/benchmark_results_dataset_b.json`

Both benchmark result files contain 12 queries and results for all four retrieval methods.

### Dense indexes

- Dataset A: `indexes/dense_index_dataset_a.npz`
- Dataset B: `indexes/dense_index_dataset_b.npz`

The older `indexes/dense_index.npz` file is not used by the separate A/B benchmark configuration.

## 7. How to reproduce the pipeline

Run these commands from the `Week2` project root in PowerShell, with the project virtual environment activated.

```powershell
python -m pip install -r requirements.txt
python src/ocr_d4.py
python src/week3_datasets.py
python src/dense_retrieval.py
python src/benchmark.py
python src/benchmark_dataset_b.py
```

`dense_retrieval.py` is interactive: select Dataset A or Dataset B when prompted. The benchmark scripts generate the main A/B result files.

The OCR script requires Tesseract to be installed. In the current Windows setup, its executable path is configured in `src/ocr_d4.py` as:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

This path may need to be changed on another computer if Tesseract is installed elsewhere.

## 8. Python dependencies

The project's `requirements.txt` contains:

```text
numpy
sentence-transformers
rank-bm25
scikit-learn
torch
PyMuPDF
pandas
tqdm
opencv-python
pytesseract
```

Tesseract OCR is a separate system application and is not installed by the Python requirements file.

## 9. Validation completed

The following checks were completed:

- Dataset A contains the four original documents.
- Dataset B contains Dataset A plus nine additional OCR documents.
- D4 is included in both datasets.
- No empty pages were reported during dataset validation.
- Chunk IDs are unique within each dataset.
- Chunk metadata preserves document, source-file, and page traceability.
- Dataset A chunk IDs are a subset of Dataset B chunk IDs.
- Separate dense index files exist for Dataset A and Dataset B.
- Both benchmark scripts ran successfully.
- Both benchmark result files contain 12 queries and all four retrieval methods.
- Retrieved results include document/page/chunk traceability fields.
- The Python dependency and import checks completed successfully in the current environment.

## 10. Member 3 deliverables

The Member 3 handover includes:

1. Dataset A and Dataset B page and chunk files.
2. Separate dense indexes for Dataset A and Dataset B.
3. BM25, dense, hybrid RRF, and hybrid-plus-reranking retrieval implementations.
4. Benchmark scripts for both datasets.
5. Benchmark result JSON files with retrieval ranks, scores, and source traceability.
6. This README documenting the pipeline and reproduction steps.

**Scope note:** This document covers retrieval pipeline implementation and indexing. Retrieval-quality metric interpretation and plots/visual analysis are handled by the respective team members responsible for evaluation and analysis.
