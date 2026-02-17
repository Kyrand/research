# Research Notes: Kleister Charity Playgroup Preparation

## 2026-02-17: Goal

Research the Kleister Charity dataset and current approaches for LLM-based fact extraction from scanned PDFs, to prepare suggested approaches for a playgroup session.

## 2026-02-17: Dataset Analysis

### Source
- GitHub: https://github.com/applicaai/kleister-charity
- Paper: https://arxiv.org/abs/2105.05796 (ICDAR 2021)

### Dataset structure
- **2,788 documents**, 61,643 pages, 21,612 entities
- Splits: train (1,729), dev (440), test (609)
- Average ~22 pages per document
- UK Charity Commission annual reports — printed and scanned
- Mix of born-digital and scanned PDFs

### 8 fields to extract
1. `address__post_town` (uppercase)
2. `address__postcode` (uppercase)
3. `address__street_line`
4. `charity_name` (uppercase)
5. `charity_number`
6. `income_annually_in_british_pounds` (2 decimal places, e.g. `1000000.00`)
7. `report_date` (ISO 8601: `YYYY-MM-DD`)
8. `spending_annually_in_british_pounds` (2 decimal places)

### Input format (TSV with 6 columns)
1. `filename` — MD5 hash
2. `keys` — space-separated list of fields to predict (alphabetical)
3. `text_djvu` — pdf2djvu/djvu2hocr output
4. `text_tesseract` — Tesseract 4.1.1 output
5. `text_textract` — Amazon Textract output
6. `text_best` — combined (uses djvu unless tesseract exceeds by 1000+ chars)

### Gold standard format (expected.tsv)
Space-separated `key=value` pairs, alphabetically sorted. Spaces/colons in values replaced with underscores.
Example: `address__post_town=LONDON address__postcode=SW1A_1AA charity_name=EXAMPLE_CHARITY ...`

### Important: "decoy keys" — some keys have no value and should return nothing.

### Evaluation metric: F1 on upper-cased values (per-entity breakdown available via GEval)

## 2026-02-17: Known Baselines (from the paper)

| Model | F1 (%) |
|-------|--------|
| Flair | 79.18 |
| BERT-base | 76.28 |
| RoBERTa-base | 81.53 |
| LayoutLM-base | 80.02 |
| LAMBERT-base | **83.57** |

LAMBERT's biggest gains over text-only models: income (+4.03) and spending (+5.60) — these appear in tables where layout matters.

Per-entity difficulty:
- Easiest: `charity_number`, `report_date` (distinctive format, sequential context)
- Hardest: `income`, `spending` (~5% require scale inference — "thousands"/"millions"), address fields (multiple addresses per doc), `charity_name` (acronyms, variations)

## 2026-02-17: Key challenge factors

1. Long documents (avg ~22 pages) — exceeds typical transformer context
2. Complex layouts (tables, multi-column)
3. Mix of scanned and born-digital PDFs
4. Scale inference for monetary values (~5%)
5. Multiple addresses present — must identify the *charity's* address
6. OCR quality varies across tools
7. Decoy keys (fields that should return nothing)
8. Best ML result (83.57%) far below simpler benchmarks (SROIE 98.17%)

## 2026-02-17: LLM approaches researched

### Finding: specialized models still beat general LLMs
- Indico Data 2024 leaderboard: fine-tuned RoBERTa/DeBERTa > all LLMs on structured extraction
- Claude V2 achieved only F1 ~0.38 on NDA extraction vs RoBERTa/DeBERTa >0.8
- But: newer models (GPT-4o, Claude 3.5+, Gemini 2.0) are much better, and the gap is closing

### Key insight: schema field naming matters dramatically
- Instructor team found changing one field name (`final_choice` → `answer`) improved accuracy from 4.5% to 95%
- Field naming can matter more than model choice

### Vision vs text-based approaches
- Vision: simpler pipeline, sees layout/tables naturally, but more expensive per-token
- Text: cheaper, works well for clean text, but loses layout information
- Hybrid: send both image + OCR text — best of both worlds

### Best tools for rapid prototyping
- **Instructor** (Python): structured extraction with Pydantic validation, works with OpenRouter
- **Zerox**: vision-based extraction, handles PDF→image→LLM pipeline
- **PyMuPDF**: fast PDF→image conversion (no external deps)
- **Docling** (IBM): layout-aware document conversion

### Cost estimates for ~100 pages
- Gemini 2.0 Flash: ~$0.02
- GPT-4o-mini: ~$0.10-0.50
- GPT-4o: ~$2-5
- Free OpenRouter models (Qwen3-VL, etc.): $0

## 2026-02-17: Approach suggestions developed

Settled on 4 main approaches for the playgroup, organized by complexity and time investment. See README.md for the full write-up.
