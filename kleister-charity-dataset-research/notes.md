# Kleister Charity Dataset Research - Notes

## 2026-02-17: Starting research

### Goals
- Understand the Kleister Charity dataset structure from the GitHub repo
- Analyze the associated paper (arxiv 2105.05796)
- Find recent work and approaches on this dataset

### Sources to investigate
1. GitHub: https://github.com/applicaai/kleister-charity
2. Paper: https://arxiv.org/abs/2105.05796
3. Web search for recent approaches

---

## 2026-02-17: GitHub Repo Analysis

Successfully fetched the full README from the GitHub repo. Key findings:

### Repository Structure
- `README.md`, `config.txt`, `in-header.tsv`
- `train/` (in.tsv.xz + expected.tsv), `dev-0/` (same), `test-A/` (same)
- `documents/` directory (~12GB via git-annex)
- Shell scripts for downloading documents from S3

### Field Names (8 entities to extract)
1. `address__post_town` (uppercase)
2. `address__postcode` (uppercase)
3. `address__street_line`
4. `charity_name` (uppercase)
5. `charity_number`
6. `income_annually_in_british_pounds`
7. `report_date`
8. `spending_annually_in_british_pounds`

### Input Format (6 TSV columns)
Column headers from in-header.tsv: `filename`, `keys`, `text_djvu`, `text_tesseract`, `text_textract`, `text_best`

### Config.txt (GEval)
Complex metric config with per-entity metrics including F1, Precision, Recall for each entity type.

### Important: "decoy keys" may be present (no value should be given for them).

---

## 2026-02-17: Paper Research (arxiv 2105.05796)

**Title:** "Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts"
**Authors:** Tomasz Stanislawek, Filip Gralinski, Anna Wroblewska, Dawid Lipinski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, Przemyslaw Biecek
**Published:** ICDAR 2021 (Springer LNCS)

### Dataset Statistics
- Kleister Charity: 2,788 documents, 61,643 unique pages, 21,612 entities
- Kleister NDA: 540 documents, 3,229 unique pages, 2,160 entities

### Split Sizes
- Train: 1,729 items
- Dev-0: 440 items
- Test-A: 609 items
- Split via stable pseudorandom hashes of document contents

### Pipeline Approach
1. Autotagging (regexp-based mechanism to create training labels)
2. Standard NER (sequence labeling with various models)
3. Text normalization
4. Final selection of entity values

### Overall Baseline Results (Kleister Charity, F1%)
- Flair: ~79.18
- BERT-base: ~76.28
- RoBERTa-base: ~81.53
- LayoutLM-base: ~80.02
- LAMBERT-base: **83.57** (best)

### Per-Entity Observations
- charity_number and report_date: highest F1 (sequential context, easier)
- LAMBERT biggest improvement over sequential models: income (+4.03) and spending (+5.60)
- income/spending: ~5% of cases require scale inference (thousand/million)
- post_town, postcode, street_line, charity_name: harder entities
- Human performance measured on 100 random documents (annotator agreement)
- 4 annotators annotated 60 documents on text span level

### Key Finding
LAMBERT (Layout-Aware Language Model using BERT with 2D positional features) proved importance of layout information. Improvement of 2.04 F1 points over best text-only model.

---

## 2026-02-17: DUE Benchmark Integration

Kleister Charity is part of the DUE (Document Understanding Evaluation) benchmark (NeurIPS 2021).
- DUE uses T5 and T5+2D as baselines
- T5+2D adds horizontal and vertical positional bias
- Challenge: encoder-decoder model can't process full document (memory limits)
- DUE supports three OCR engines: microsoft_cv, tesseract, djvu

---

## 2026-02-17: Recent Work and LLM Approaches

### Indico Data LLM Leaderboard (2024)
- Tested LLMs (Llama, Azure OpenAI, Google, AWS Bedrock) on document tasks including Kleister Charity
- Finding: Indico's own discriminative models (RoBERTa, DeBERTa) outperformed LLMs on extraction
- Claude V2 scored only 0.38 F1 on NDA extraction vs. RoBERTa/DeBERTa >0.8
- LLMs better at summarization; worse at structured extraction

### LayoutLMv2/v3
- LayoutLMv2 showed SOTA on Kleister-NDA (0.834 -> 0.852)
- No published Kleister-Charity specific LayoutLMv2/v3 results found
- Community fine-tuned models on HuggingFace exist for Kleister-NDA but not Charity

### TILT (Text-Image-Layout Transformer)
- Developed by same team (Applica AI)
- End-to-end model with decoder for layout understanding
- Evaluated mainly on DocVQA, CORD, SROIE

### Pix2Struct
- OCR-free approach, pixel-to-text
- Sensitive to hyperparameters, mixed practical results

### Papers with Code
- Was shut down by Meta in July 2025
- Historical leaderboard data may be available in archives

### Key Trend
- Document AI models (LayoutLM family, LAMBERT, TILT) outperform generic LLMs on structured extraction
- LLMs face challenges with spatial/layout-dependent tasks
- Cost-prohibitive for large-scale document processing
- Zero-shot LLM extraction is flexible but inconsistent

---

## 2026-02-17: Summary of Challenges

1. Long documents (avg ~22 pages per document)
2. Mix of scanned and born-digital PDFs
3. Complex layouts (tables, multi-column)
4. Scale inference for monetary values (~5% of cases)
5. Multiple addresses present (must identify charity address specifically)
6. OCR quality varies across extraction tools
7. Entity values not always directly stated (may need inference)
8. Decoy keys with no corresponding value
9. Gap to human performance is significant
10. Best ML result (83.57%) far below simpler benchmarks (e.g., SROIE 98.17%)
