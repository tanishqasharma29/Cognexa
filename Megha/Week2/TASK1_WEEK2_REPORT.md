## Department of Management Studies, ABV-IIITM Gwalior REPORT

## Trustworthy AI Project – Semester Project

## Task 1 – Retrieval Evaluation and Dataset Inventory

## Corpus Expansion (Datasets A & B), Ground-Truth Mapping, and Precision/Recall/MRR Evaluation

## Task 1 Technical Report

Submitted by

## Megha Saini

## Task 1 Lead – Dataset Inventory, Ground-Truth Curation, Precision@K, Recall@K, and MRR Evaluation

## Under the Guidance of

## Dr. Roshni Chakraborty

## Atal Bihari Vajpayee Indian Institute of Information Technology and Management, Gwalior

November 2026


## Contents

## Key Findings at a Glance

| Scope | Evaluation of four retrieval methods (BM25, dense, hybrid, hybrid + reranking) on Dataset A |
| --- | --- |
|   | (19,964 selectable-text chunks) and Dataset B (22,261 chunks, including 2,297 OCR-derived |
|   | chunks), using a fixed 19-query benchmark (9 common + 10 OCR-specific). |
| Best method | Hybrid + reranking gave the highest scores on Dataset B: P@1 = 0.737, R@3 = 0.895 and MRR = |
|   | 0.807 (+124.2% over its Dataset A MRR of 0.360). |
| Effect of OCR | All four methods improved on Dataset B. BM25 MRR rose from 0.333 to 0.644 (+93.4%), dense |
| chunks | from 0.189 to 0.330 (+74.6%), hybrid from 0.303 to 0.606 (+100.0%). BM25 Recall@10 rose from |
|   | 0.368 to 0.842. |
| Speed | Hybrid + Reranking is the slowest at roughly 1.1 to 1.2 s per query, against about 0.2 s for hybrid |
|   | and BM25; adding OCR chunks did not noticeably change retrieval time. |
| Corpus growth | Pages increased from 18,477 to 20,683 (+11.9%) and chunks from 19,964 to 22,261 (+11.5%) |
|   | across the same 1,267 PDFs. |
| Main caveat | The 19-query suite includes 10 OCR-specific queries, so Dataset A scores partly reflect content that |
|   | is absent from that corpus. |


## Task 1 – Retrieval Evaluation and Dataset Inventory

## Dataset Inventory, Benchmark Query Curation, and Retrieval Evaluation on Two Corpora

This report covers the Task 1 contribution to the project: ownership of the benchmark datasets (Dataset A and Dataset B), their structural documentation, grounded query curation, and a comparative retrieval evaluation. It compares lexical, dense, hybrid and reranked retrieval on the same benchmark queries and measures how expanding the corpus with OCR-derived text changes the results.

## 1. Objective

As lead for Task 1, the mandate covered the complete lifecycle of the benchmark datasets: documenting the baseline collection (Dataset A), expanding it with representative scanned and handwritten PDF records (Dataset B), curating ground-truth mappings, and evaluating retrieval with Precision@K, Recall@K and MRR. Specifically, the executed tasks included:

- Dataset Inventory Ownership: documenting the baseline document collection (Dataset A) and expanding it by introducing representative scanned and handwritten PDF records into the corpus (Dataset B).

- Document Profiling: tracking Document ID, subject matter, structural document type, page counts, file size, selectable vs. scanned status, and institutional source repository.

- Ground-Truth Passage Mapping: identifying and cataloging the precise pages, paragraphs and chunks containing authoritative ground-truth answers for each benchmark query.

- Fixed Query Protocol: establishing a controlled, standardized query suite so that identical queries are tested pre- and post-OCR addition without bias.

## 2. Evaluation Pipeline and Settings

Retrieval was benchmarked across four standard architectural configurations on both datasets, using the same 19 queries, so that any difference between datasets comes from the corpus and not from the query set.

```
Benchmark queries (19) + ground-truth mapping
-> Dataset A corpus (19,964 chunks) | Dataset B corpus (22,261 chunks, incl. 2,297 OCR chunks)
-> BM25 | dense | hybrid (RRF) | hybrid + cross-encoder reranking
-> ranked chunks
-> Precision@K, Recall@K (K = 1, 3, 5, 10), MRR
-> Dataset A vs B comparison -> charts -> report
```

| Component | Setting used |
| --- | --- |
| Document collection | 1,267 PDFs (same in both datasets) |
| Retrieval methods | Lexical (BM25); Dense vector embeddings; Hybrid fusion via Reciprocal Rank Fusion |
|   | (RRF); Hybrid + Cross-Encoder Reranking |
| Cut-offs | K = 1, 3, 5 and 10 for Precision and Recall |
| Ranking metric | Mean Reciprocal Rank (MRR) |
| Benchmark suite | 19 queries: 9 common + 10 OCR-specific; identical for Dataset A and Dataset B |

*Table A. Retrieval and evaluation settings used for both datasets.*


## 3. Component Explanations

For each evaluation component, the purpose, input, output, method, limitation and evaluation are summarised below, in the same layout as the Week 2 report.

## 3.1 Dataset inventory

| Purpose | Maintain a complete, auditable catalog of the benchmark corpus in its two states (Dataset A and |
| --- | --- |
|   | Dataset B). |
| Input | 1,267 PDFs: natively digital, selectable-text documents plus representative scanned and handwritten |
|   | records. |
| Output | Dataset Inventory file with Document ID, subject, document type, page count, file size, selectable vs. |
|   | scanned status and source repository. |
| Method | Dataset A is the baseline of selectable-text PDFs; Dataset B adds OCR ingestion of scanned and |
|   | handwritten documents on top of it. |
| Limitation | Dataset B adds OCR-derived text only; quality of that text depends on the scan and handwriting |
|   | quality, which is recorded as OCR quality notes. |
| Evaluation | Counts reconciled in Table 1: 19,964 selectable-text chunks + 2,297 OCR-derived chunks = 22,261 |
|   | chunks. |

## 3.2 Handwritten and scanned document registry

| Purpose | Record exactly which scanned/handwritten documents were added to create Dataset B. |
| --- | --- |
| Input | Representative scanned and handwritten PDF records. |
| Output | Granular register with Document IDs, subjects, page dimensions, physical sizes and OCR quality |
|   | notes. |
| Method | Each added document is logged individually so that OCR-derived chunks can be traced to their |
|   | source. |
| Limitation | The register describes the added documents; it does not itself measure retrieval quality. |
| Evaluation | Reflected in the page growth from 18,477 to 20,683 (+2,206 pages). |

## 3.3 Ground-truth mapping and fixed query protocol

| Purpose | Provide a fixed set of questions with known authoritative answers so retrieval can be scored |
| --- | --- |
|   | objectively. |
| Input | 19 benchmark queries: 9 common queries and 10 OCR-specific queries. |
| Output | For each query, the explicit passage, chunk ID and page references containing the ground-truth |
|   | answer. |
| Method | The same 19 queries and the same mapping are used for Dataset A and Dataset B, so the two runs are |
|   | directly comparable. |
| Limitation | The set is small: one query moves Precision@1 and Recall@1 by 1/19 (about 0.053). |
| Evaluation | Validated benchmark mapping with explicit passage, chunk ID and page references. |


## 3.4 Precision@K

| Purpose | Measure how much of the top K retrieved results is relevant. |
| --- | --- |
| Input | Ranked chunk list and the ground-truth mapping for each query. |
| Output | Relevant chunks in the top K divided by K, averaged over the 19 queries. |
| Method | Computed at K = 1, 3, 5 and 10. |
| Limitation | Because K is the denominator, Precision falls as K grows when only a few chunks are relevant per |
|   | query. |
| Evaluation | Reported in Table 2. |

## 3.5 Recall@K

| Purpose | Measure how much of the relevant evidence is found within the first K results. |
| --- | --- |
| Input | Ranked list and the ground-truth mapping for each query. |
| Output | Relevant chunks retrieved in the top K divided by total relevant chunks, averaged over the 19 queries. |
| Method | Computed at K = 1, 3, 5 and 10. |
| Limitation | Cannot rise for queries whose answer is not present in the corpus, which caps Recall on Dataset A. |
| Evaluation | Reported in Table 3 and Figure 1. |

## 3.6 Mean Reciprocal Rank (MRR)

| Purpose | Reward methods that place the first relevant chunk near the top. |
| --- | --- |
| Input | Rank r of the first relevant chunk for each query. |
| Output | Mean of 1/r across the 19 queries. |
| Method | A first relevant result at rank 1 contributes 1.0, at rank 2 contributes 0.5, and so on. |
| Limitation | Considers only the first relevant result and ignores any others. |
| Evaluation | Reported in Table 4 and Figures 3 and 4. |

## 3.7 Average retrieval time per query

| Purpose | Quantify the speed cost of each retrieval configuration. |
| --- | --- |
| Input | Recorded retrieval time per query for the 9 common queries. |
| Output | Mean time in seconds per method, before OCR (Dataset A) and after OCR (Dataset B). |
| Method | Each method is timed on the same 9 common queries on both datasets. |
| Limitation | Covers the 9 common queries only, not the full 19-query suite. |
| Evaluation | Reported in Figure 5. |

## 4. Datasets

Dataset A represents the baseline repository of natively digital, selectable-text PDFs. Dataset B expands this baseline through optical character recognition (OCR) ingestion of scanned and handwritten documentation, scaling overall volume while maintaining structural integrity.


| Metric / Attribute | Dataset A (Without | Dataset B (With OCR) | Net Change |
| --- | --- | --- | --- |
|   | OCR) |   |   |
| Total PDFs Processed | 1,267 | 1,267 | 0 |
| Total Document Pages | 18,477 | 20,683 | +2,206 (+11.9%) |
| Total Ingested Chunks | 19,964 | 22,261 | +2,297 (+11.5%) |
| Selectable-Text Chunks | 19,964 | 19,964 | 0 |
| OCR-Derived Chunks | 0 | 2,297 | +2,297 |
| Evaluation Queries (Overall | 19 | 19 | Controlled Set |
| Benchmark) |   |   |   |

*Table 1. Corpus inventory and structural comparison, Dataset A and Dataset B.*

## 5. Benchmark Queries

Nineteen queries were run against each dataset: 9 common queries and 10 OCR-specific queries. Every query has a validated ground-truth mapping with explicit passage, chunk ID and page references. Precision, Recall and MRR are all averaged over the full 19-query suite, and the identical suite is used for Dataset A and Dataset B.

## 6. Results

Results are reported over the complete 19-query benchmark for all four retrieval architectures, comparing Dataset A (Without OCR) and Dataset B (With OCR).

## 6.1 Precision@K

| Method | Dataset | P@1 | P@3 | P@5 | P@10 |
| --- | --- | --- | --- | --- | --- |
| BM25 | A (Without OCR) | 0.316 | 0.123 | 0.074 | 0.037 |
| BM25 | B (With OCR) | 0.579 | 0.228 | 0.147 | 0.084 |
| Dense | A (Without OCR) | 0.105 | 0.088 | 0.063 | 0.032 |
| Dense | B (With OCR) | 0.158 | 0.158 | 0.116 | 0.063 |
| Hybrid | A (Without OCR) | 0.263 | 0.105 | 0.063 | 0.042 |
| Hybrid | B (With OCR) | 0.526 | 0.228 | 0.137 | 0.090 |
| Hybrid + Reranking | A (Without OCR) | 0.316 | 0.140 | 0.084 | 0.042 |
| Hybrid + Reranking | B (With OCR) | 0.737 | 0.298 | 0.179 | 0.090 |

*Table 2. Precision@K by method and dataset (19 queries).*


## 6.2 Recall@K

| Method | Dataset | R@1 | R@3 | R@5 | R@10 |
| --- | --- | --- | --- | --- | --- |
| BM25 | A (Without OCR) | 0.316 | 0.368 | 0.368 | 0.368 |
| BM25 | B (With OCR) | 0.579 | 0.684 | 0.737 | 0.842 |
| Dense | A (Without OCR) | 0.105 | 0.263 | 0.316 | 0.316 |
| Dense | B (With OCR) | 0.158 | 0.474 | 0.579 | 0.632 |
| Hybrid | A (Without OCR) | 0.263 | 0.316 | 0.316 | 0.421 |
| Hybrid | B (With OCR) | 0.526 | 0.684 | 0.684 | 0.895 |
| Hybrid + Reranking | A (Without OCR) | 0.316 | 0.421 | 0.421 | 0.421 |
| Hybrid + Reranking | B (With OCR) | 0.737 | 0.895 | 0.895 | 0.895 |

*Table 3. Recall@K by method and dataset (19 queries).*

## 6.3 Mean Reciprocal Rank

| Method | Without OCR (Dataset A) | With OCR (Dataset B) | Net Improvement |
| --- | --- | --- | --- |
| BM25 | 0.333 | 0.644 | +0.311 (+93.4%) |
| Dense | 0.189 | 0.330 | +0.141 (+74.6%) |
| Hybrid | 0.303 | 0.606 | +0.303 (+100.0%) |
| Hybrid + Reranking | 0.360 | 0.807 | +0.447 (+124.2%) |

*Table 4. MRR by method and dataset (19 queries).*

## 6.4 BM25 benchmark visualizations

The following charts illustrate the 19-query benchmark evaluation for BM25, contrasting performance Without OCR (Dataset A) and With OCR (Dataset B). Adding OCR text enables the retrieval system to resolve OCR-dependent questions, driving sharp gains across Recall (R@1: 0.316 to 0.579; R@10: 0.368 to 0.842), Precision (P@1: 0.316 to 0.579) and MRR (0.3333 to 0.6444).

## BM25 Recall by cut-off queries)

*Figure 1. BM25 – Recall comparison across R@1, R@3, R@5, R@10 (19 queries).*


## BM25 - Precision by cut-off (19 queries)

*Figure 2. BM25 – Precision comparison across P@1, P@3, P@5, P@10 (19 queries).*

*Figure 3. BM25 – Mean Reciprocal Rank comparison (0.3333 vs. 0.6444).*

## Mean Reciprocal Rank (MRR) by method

*Figure 4. MRR per retrieval method on Dataset A and Dataset B (19 queries).*

## 6.5 Average retrieval time per query

Retrieval time was also compared before and after OCR for the 9 common queries. BM25 and dense retrieval are the fastest, hybrid is slightly slower, and Hybrid + Reranking is clearly the slowest at roughly 1.1 to 1.2 s per query. Adding the OCR chunks left the time for every method essentially unchanged.


## Average retrieval time per query (9 common queries)

*Figure 5. Average retrieval time per query for the 9 common queries, before and after OCR (values read approximately from*

*the chart).*

## 7. Dataset A vs Dataset B Comparison

Table 5 shows the change from Dataset A to Dataset B (B minus A), computed from Tables 2 to 4. A positive change is an increase in the metric.

| Method | R@1 | R@3 | R@5 | R@10 | P@1 | P@3 | P@5 | P@10 MRR |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | +0.263 | +0.316 | +0.369 | +0.474 | +0.263 | +0.105 | +0.073 | +0.047 | +0.311 |
| Dense | +0.053 | +0.211 | +0.263 | +0.316 | +0.053 | +0.070 | +0.053 | +0.031 | +0.141 |
| Hybrid | +0.263 | +0.368 | +0.368 | +0.474 | +0.263 | +0.123 | +0.074 | +0.048 | +0.303 |
| Hybrid + Reranking | +0.421 | +0.474 | +0.474 | +0.474 | +0.421 | +0.158 | +0.095 | +0.048 | +0.447 |

*Table 5. Change from Dataset A to Dataset B (B A). R = Recall, P = Precision.*

## 8. Observations and Interpretation

- Substantial gain from OCR text ingestion. In Dataset A, queries targeting scanned content failed due to the absence of textual representations, capping Recall@10 at 0.368 for BM25 and 0.421 for Hybrid + Reranking. Incorporating OCR chunks in Dataset B boosted Recall@10 to 0.842 for BM25 and 0.895 for Hybrid + Reranking across the entire 19-query suite.

- Peak ranking power of Hybrid + Reranking. Hybrid + Reranking achieved the top performance on Dataset B, delivering P@1 of 0.737, R@3 of 0.895 and an MRR of 0.807 (a +124.2% jump over its Dataset A baseline).

- BM25 keyword sensitivity. Lexical BM25 responded strongly when OCR text was introduced, with MRR rising from 0.333 to 0.644 and P@1 from 0.316 to 0.579, confirming that keyword extraction from handwritten and scanned files effectively bridges the information retrieval gap.

- Speed versus quality. Hybrid + Reranking is the slowest method (roughly 1.1 to 1.2 s per query, against about 0.2 s for hybrid and BM25 and under 0.05 s for dense), which is the price of its best ranking quality. Adding OCR chunks did not noticeably change retrieval time for any method (Figure 5).

- Dense retrieval gains least. Dense retrieval improved (MRR 0.189 to 0.330, R@10 0.316 to 0.632) but remained the lowest-scoring method on both datasets across all MRR, Precision@1 and Recall@1 figures.


## 9. Limitations

- The benchmark is small: 19 queries, so a single query shifts Precision@1 and Recall@1 by about 0.053. Small differences between methods should not be over-interpreted.

- The 19-query suite contains 10 OCR-specific queries, so Dataset A scores partly reflect content that Dataset A does not contain; the A vs B gains should be read with this in mind.

- Retrieval time (Figure 5) covers only the 9 common queries, and its values are read approximately from the chart rather than from a table.

## 10. Recommendations

- Use Hybrid + Reranking as the default retrieval configuration, as it gave the best P@1, Recall@3 and MRR on Dataset B.

- Retain the OCR-derived chunks in the corpus, since every method improved on Dataset B.

- Keep plain hybrid retrieval as a lower-latency fallback if response time becomes a constraint.

- Expand the query set so that individual queries have less influence on each metric.

## 11. Completed Task Deliverables

| File / Deliverable | Description |
| --- | --- |
| Dataset Inventory File | Comprehensive audit log detailing 1,267 documents, page distributions (18,477 |
|   | baseline to 20,683 expanded), and chunk splits (19,964 selectable vs. 2,297 OCR). |
| Handwritten & Scanned | Granular register containing Document IDs, subjects, page dimensions, physical sizes, |
| Document Log | and OCR quality notes. |
| Benchmark Query & | Validated benchmark mapping containing 19 queries (9 common + 10 OCR-specific) |
| Ground-Truth Reference | with explicit passage, chunk ID, and page references. |

*Table 6. Task 1 deliverables.*

## 12. Conclusion

On the 19-query benchmark, adding OCR-derived text (Dataset B) improved every retrieval method. Hybrid

\+ Reranking provides the strongest overall ranking quality on Dataset B, with P@1 of 0.737, Recall@3 of 0.895 and MRR of 0.807, while BM25 shows the sharpest response to the added OCR text, with MRR rising from 0.333 to 0.644. These results support the inclusion of OCR ingestion in the corpus and Hybrid + Reranking as the strongest of the four tested configurations for this query set.
