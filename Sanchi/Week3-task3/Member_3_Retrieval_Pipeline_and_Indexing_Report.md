# MEMBER 3 REPORT --- Retrieval Pipeline and Indexing

**Project:** Cognexa -- RAG Quiz Platform\
**Role:** Member 3 --- Retrieval Pipeline and Indexing

------------------------------------------------------------------------

## 1. Introduction

This report documents the retrieval pipeline and indexing work completed
for the Cognexa RAG Quiz Platform. The work focuses on preparing
separate retrieval datasets, generating dense vector indexes,
implementing lexical and hybrid retrieval, applying reranking, and
exporting query-level retrieval results for comparison.

The experiment compares retrieval on the original extracted PDF
collection (Dataset A) with retrieval after adding nine additional
handwritten/scanned OCR documents (Dataset B).

## 2. Objectives

-   Prepare separate Dataset A and Dataset B retrieval collections.
-   Apply the same chunking and embedding configuration to both
    datasets.
-   Build separate dense indexes for both datasets.
-   Run BM25, dense, hybrid Reciprocal Rank Fusion (RRF), and
    hybrid-plus-reranking retrieval.
-   Save query-level top-K results with scores, ranks, and source
    metadata.
-   Preserve OCR/original-document identification so retrieved chunks
    can be traced to their sources.

## 3. Dataset Preparation and Statistics

Dataset A contains the successfully extracted original College and
Kaggle PDF documents. Dataset B contains the same original collection
plus nine additional OCR-processed documents.

  Dataset detail                                    Dataset A   Dataset B
  ----------------------------------------------- ----------- -----------
  Source PDFs discovered                                1,267       1,267
  Successfully extracted original PDF documents         1,257       1,257
  Additional OCR documents                                  0           9
  Total documents represented                           1,257       1,266
  Extracted pages                                      20,830      20,969
  Chunks                                               20,646      20,785

**Extraction note:** One encrypted PDF and nine zero-page PDFs could not
be extracted. Therefore, 1,257 original PDF documents were successfully
represented in both datasets. Dataset B adds nine OCR documents
containing 139 OCR-processed pages/chunks.

### Source composition

  Source             Dataset A chunks   Dataset B chunks
  ---------------- ------------------ ------------------
  College                       6,888              6,888
  Kaggle                       13,758             13,758
  Additional OCR                    0                139
  **Total**                **20,646**         **20,785**

## 4. Chunking Configuration

The same chunking configuration was used for Dataset A and Dataset B to
keep the comparison consistent.

  Parameter          Configuration
  ------------------ ------------------------------------
  Chunk size         700 words
  Chunk overlap      100 words
  Dataset A output   `data/processed/dataset_A_actual/`
  Dataset B output   `data/processed/dataset_B_actual/`

## 5. Dense Embedding and Indexing

The embedding model used for both datasets was `all-MiniLM-L6-v2`,
producing 384-dimensional embeddings.

  ----------------------------------------------------------------------------------------------------
  Dataset               Number of chunks Embedding shape  Index file
  ---------------- --------------------- ---------------- --------------------------------------------
  Dataset A                       20,646 `(20646, 384)`   `indexes/dense_index_actual_dataset_a.npz`

  Dataset B                       20,785 `(20785, 384)`   `indexes/dense_index_actual_dataset_b.npz`
  ----------------------------------------------------------------------------------------------------

The saved index chunk IDs were checked against the corresponding chunks
file to ensure that embedding rows remained aligned with chunk metadata.

## 6. Retrieval Methods

Four retrieval methods were executed for the benchmark queries.

### 6.1 BM25 Lexical Retrieval

BM25 retrieves documents using lexical term matching. A separate BM25
corpus was built for each dataset.

### 6.2 Dense Retrieval

Dense retrieval embeds the query using `all-MiniLM-L6-v2` and compares
it with the saved chunk embeddings to retrieve semantically similar
chunks.

### 6.3 Hybrid Retrieval with Reciprocal Rank Fusion

Hybrid retrieval combines dense and BM25 rankings using Reciprocal Rank
Fusion.

The RRF formula is:

\[ RRF(d) = `\sum`{=tex}\_{r `\in `{=tex}R(d)}
`\frac{1}{k + rank_r(d)}`{=tex} \]

where `rank_r(d)` is the rank of document/chunk `d` in a retrieval
result list and `k` is the RRF constant.

**RRF constant:** `k = 60`.

### 6.4 Hybrid Retrieval with Cross-Encoder Reranking

The hybrid retrieval candidates are reranked using:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The cross-encoder evaluates the relevance of each query--chunk pair and
produces a reranking score.

## 7. Retrieval Configuration

  Parameter                           Value
  ----------------------------------- ----------------------------------------
  Embedding model                     `all-MiniLM-L6-v2`
  Embedding dimensions                384
  Reranker                            `cross-encoder/ms-marco-MiniLM-L-6-v2`
  Candidate K                         10
  RRF K                               60
  Exported results per method/query   10
  Benchmark queries per dataset       12
  Retrieval methods                   4

The same model and retrieval settings were used for Dataset A and
Dataset B.

## 8. Query-Level Retrieval Exports

Twelve benchmark queries were executed on each dataset using all four
retrieval methods:

1.  Lexical (BM25)
2.  Dense
3.  Hybrid (RRF)
4.  Hybrid with reranking

Each query-method combination contains ten ranked results.

  Output                           Dataset A   Dataset B
  ------------------------------ ----------- -----------
  Queries                                 12          12
  Methods                                  4           4
  Results per query and method            10          10
  Total ranked entries                   480         480

**Total exported ranked retrieval entries across both datasets: 960.**

### Output files

-   `results/actual_retrieval_dataset_a.json`
-   `results/actual_retrieval_dataset_b.json`

These JSON files preserve the query-level results rather than only
aggregate averages.

## 9. Result Metadata and Traceability

Each retrieved result includes identifying and ranking information,
including:

-   Query ID and query text
-   Dataset identifier
-   Chunk ID
-   Document ID
-   Source filename and source path
-   Page number
-   Source dataset/category
-   Extraction method
-   Retrieval rank
-   Retrieval score and applicable method-specific ranking information

This metadata allows each retrieved chunk to be traced back to its
document and page.

### OCR and original-document identification

The `dataset` and `extraction_method` fields distinguish the source of a
chunk:

  Source type                  Dataset metadata
  ---------------------------- --------------------
  Original College documents   `college`
  Original Kaggle documents    `kaggle`
  Additional OCR documents     `ocr`
  OCR extraction method        `ocr_preprocessed`

This makes it possible to identify whether a result came from an
original extracted PDF or one of the additional OCR documents.

## 10. Validation Performed

The following checks were completed on the exported retrieval results:

  ----------------------------------------------------------------------------
  Validation                          Result
  ----------------------------------- ----------------------------------------
  Dataset A chunk count               20,646

  Dataset B chunk count               20,785

  Query count in each dataset         12

  Results per method/query            10

  Total ranked entries across both    960
  datasets                            

  Missing required metadata entries   0

  Incorrect rank sequences            0

  Candidate K consistency             10 for both datasets

  RRF K consistency                   60 for both datasets

  Embedding model consistency         `all-MiniLM-L6-v2` for both

  Reranker consistency                `cross-encoder/ms-marco-MiniLM-L-6-v2`
                                      for both

  Python dependency check             No broken requirements found
  (`pip check`)                       
  ----------------------------------------------------------------------------

The checks confirm that the exported result files contain the expected
number of query-level ranked entries and the required traceability
metadata.

## 11. Deliverables

### Dataset and index outputs

-   `data/processed/dataset_A_actual/pages.json`
-   `data/processed/dataset_A_actual/chunks.json`
-   `data/processed/dataset_B_actual/pages.json`
-   `data/processed/dataset_B_actual/chunks.json`
-   `indexes/dense_index_actual_dataset_a.npz`
-   `indexes/dense_index_actual_dataset_b.npz`

### Retrieval scripts

-   `src/week3_datasets_actual.py`
-   `src/dense_retrieval.py`
-   `src/lexical_retrieval.py`
-   `src/hybrid_retrieval.py`
-   `src/reranking.py`
-   `src/export_actual_retrieval.py`

### Retrieval result files

-   `results/actual_retrieval_dataset_a.json`
-   `results/actual_retrieval_dataset_b.json`

## 12. Conclusion

The Member 3 retrieval and indexing work produced separate Dataset A and
Dataset B collections and dense indexes, applied consistent retrieval
configurations, and exported query-level results for BM25, dense, hybrid
RRF, and hybrid-plus-reranking methods.

Both datasets contain twelve benchmark queries, four retrieval methods,
and ten ranked results per query-method combination, giving 960 ranked
entries across the two datasets. The exported results include
document/chunk identifiers, ranks, scores, page information, source
metadata, and OCR/original-document identification.

The retrieval outputs are prepared for the evaluation stage. This report
does not claim an overall improvement in retrieval quality from adding
OCR documents; that conclusion requires the evaluation and comparison of
retrieval metrics across the benchmark queries.

------------------------------------------------------------------------

**Prepared by:** Member 3 --- Retrieval Pipeline and Indexing\
**Project:** Cognexa -- RAG Quiz Platform
