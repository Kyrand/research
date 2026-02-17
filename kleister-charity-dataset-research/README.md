# Kleister Charity Dataset: Comprehensive Research Report

## Overview

The **Kleister Charity** dataset is a Key Information Extraction (KIE) benchmark introduced by researchers at Applica AI and published at ICDAR 2021. It consists of annual financial reports from British charity organizations in PDF format, and the task is to extract structured fields such as the charity's name, address, registration number, income, and spending from these documents.

This report covers the dataset structure, the associated paper, baseline results, evaluation methodology, and recent approaches to the task.

---

## 1. GitHub Repository: applicaai/kleister-charity

**URL:** https://github.com/applicaai/kleister-charity

### 1.1 Repository Structure

```
kleister-charity/
├── README.md                          # Dataset documentation
├── config.txt                         # GEval evaluation configuration
├── in-header.tsv                      # Column header definitions for input data
├── train/
│   ├── in.tsv.xz                      # Training input (compressed TSV)
│   └── expected.tsv                   # Training ground truth
├── dev-0/
│   ├── in.tsv.xz                      # Development input (compressed TSV)
│   └── expected.tsv                   # Development ground truth
├── test-A/
│   ├── in.tsv.xz                      # Test input (compressed TSV)
│   └── expected.tsv                   # Test ground truth
├── documents/                         # PDF files (~12GB, stored via git-annex)
├── annex-get-all-from-s3.sh           # Script to download all documents from S3
└── annex-get-test-documents-from-s3.sh # Script to download test documents from S3
```

### 1.2 Data Splits

| Split   | Documents |
|---------|-----------|
| Train   | 1,729     |
| Dev-0   | 440       |
| Test-A  | 609       |
| **Total** | **2,778** |

The total reported in the paper is 2,788 documents across all splits (minor discrepancy with the sum of 2,778 likely due to filtering). The split was performed using a stable pseudorandom method based on hashes (fingerprints) of document contents to prevent organizational duplication across sets.

### 1.3 Dataset Size

- **Total documents:** 2,788 annual financial reports
- **Total unique pages:** 61,643
- **Total entities to extract:** 21,612
- **Average document length:** ~22 pages per document
- **Total PDF storage:** ~12 GB (stored via git-annex on S3)
- **Source:** UK Charity Commission (https://www.gov.uk/government/organisations/charity-commission)
- **Language:** English only

### 1.4 Fields to Extract (8 Entity Types)

| Field Name | Description | Format/Normalization |
|------------|-------------|---------------------|
| `address__post_town` | Municipal designation of the charity's address | Uppercase letters |
| `address__postcode` | Postal code of the charity | Uppercase letters |
| `address__street_line` | Street address of the charity | As-is (spaces replaced with underscores in output) |
| `charity_name` | Official name of the charitable organization | Uppercase letters |
| `charity_number` | Registered identification number | Numeric |
| `income_annually_in_british_pounds` | Annual income in GBP | Real number with two decimal places (e.g., `1000000.00`, `1590.50`) |
| `report_date` | Reporting date of the annual document | ISO 8601 format: `YYYY-MM-DD` |
| `spending_annually_in_british_pounds` | Annual expenditures in GBP | Real number with two decimal places (e.g., `230.03`) |

**Important notes:**
- Not all keys necessarily have a corresponding value in every document ("decoy keys" may be present)
- Multiple values per key are allowed (though precision decreases accordingly)
- The task is **information extraction, not NER** -- the interest is in the extracted value, not its location in the document

### 1.5 Input Format (in.tsv.xz)

The input is a compressed TSV file with 6 columns, defined in `in-header.tsv`:

| Column | Header | Description |
|--------|--------|-------------|
| 1 | `filename` | Document filename (MD5 hash + extension) |
| 2 | `keys` | Space-separated list of keys to predict (alphabetical order, underscores for word separation) |
| 3 | `text_djvu` | Plain text extracted via pdf2djvu/djvu2hocr (v0.9.8) |
| 4 | `text_tesseract` | Plain text extracted via Tesseract (v4.1.1-rc1-7-gb36c, flags: `--oem 2 -l eng --dpi 300`) |
| 5 | `text_textract` | Plain text extracted via Amazon Textract (March 1, 2020 version) |
| 6 | `text_best` | Combined: uses pdf2djvu/djvu2hocr by default, switches to tesseract if tesseract output exceeds pdf2djvu by 1000+ characters |

**Escape sequences in extracted text:**
- `\f` -- page break (form feed)
- `\n` -- line break
- `\t` -- tab character
- `\\` -- literal backslash

**Note:** End-of-lines, tabs, and non-printable characters are replaced with spaces in the text columns. The OCR text may not be perfect; users are free to use alternative OCR software.

### 1.6 Gold Standard Format (expected.tsv)

Each line corresponds to one document and contains space-separated `key=value` pairs, sorted alphabetically by key. Spaces and colons within values are replaced with underscores.

**Example line:**
```
address__post_town=LONDON address__postcode=SW1A_1AA address__street_line=10_Downing_Street charity_name=EXAMPLE_CHARITY charity_number=123456 income_annually_in_british_pounds=1000000.00 report_date=2020-03-31 spending_annually_in_british_pounds=950000.00
```

**Key rules:**
- TSV format (not CSV) -- double quotes are not special characters
- The order of key-value pairs does not matter in output
- Dates must be in `YYYY-MM-DD` (ISO 8601) format
- Monetary values must be real numbers with exactly two decimal places (dot separator)

### 1.7 Evaluation

**Primary metric:** F1 score calculated on upper-cased values
**Auxiliary metric:** F1 score on true-cased values

Evaluation uses the **GEval** tool (https://gitlab.com/filipg/geval):

```bash
wget https://gonito.net/get/bin/geval
chmod u+x geval
./geval -t dev-0
```

The `config.txt` defines per-entity evaluation metrics:
```
--metric MultiLabel-F1:uN<F1(UC)>
--metric MultiLabel-F1:N<F1>P<3>
--metric MultiLabel-F0:uN<P(UC)>P<3>
--metric MultiLabel-F9999:uN<R(UC)>P<3>
--metric Accuracy:uSN<Accuracy>P<2>
--metric Mean/MultiLabel-F1:uN<Mean-F1>P<3>
```

Per-entity metrics are computed for: `address` (combined), `money` (combined income/spending), `town`, `postcode`, `street`, `name`, `number`, `income`, `spending`, `date`.

---

## 2. The Paper: arxiv 2105.05796

### 2.1 Citation

**Title:** Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts
**Authors:** Tomasz Stanislawek, Filip Gralinski, Anna Wroblewska, Dawid Lipinski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, Przemyslaw Biecek
**Published:** ICDAR 2021 (International Conference on Document Analysis and Recognition)
**Springer:** LNCS volume 12821, pp. 564-579
**arXiv:** https://arxiv.org/abs/2105.05796
**DOI:** 10.1007/978-3-030-86549-8_36

An earlier version appeared as arXiv:2003.02356 (March 2020).

### 2.2 Paper Summary

The paper addresses the lack of well-defined benchmarks for Key Information Extraction (KIE) from complex, long documents. While existing datasets (SROIE, CORD, FUNSD) focus on short, single-page documents, Kleister introduces two datasets with multi-page documents featuring complex layouts:

1. **Kleister Charity:** 2,788 British charity annual financial reports (61,643 pages, 21,612 entities)
2. **Kleister NDA:** 540 Non-Disclosure Agreements (3,229 pages, 2,160 entities)

Both datasets involve a mix of scanned and born-digital PDFs in English, requiring systems to use both textual and structural/layout features for extraction.

### 2.3 Pipeline Approach (Baseline Method)

The authors developed a four-stage pipeline for their baselines:

1. **Autotagging:** A regexp-based mechanism that automatically labels training data at the text-span level by matching known entity values in the OCR text. This creates BIO-tagged (Begin/Inside/Outside) training sequences for NER models. The autotagging is imperfect -- it may incorrectly tag mentions or miss some.

2. **NER / Sequence Labeling:** Train a sequence labeling model on the autotagged data. Models tested: Flair, BERT, RoBERTa, LayoutLM, LAMBERT (all Base-sized).

3. **Text Normalization:** Normalize extracted spans to the expected format (ISO 8601 dates, two-decimal monetary values, uppercase where required).

4. **Final Value Selection:** Select the final entity values from the model's predictions at the document level.

### 2.4 Baseline Models Tested

| Model | Type | Key Feature |
|-------|------|-------------|
| **Flair** | Sequence labeling | Character-level contextual embeddings |
| **BERT-base** | Transformer | Bidirectional text encoder |
| **RoBERTa-base** | Transformer | Optimized BERT pretraining |
| **LayoutLM-base** | Transformer + Layout | Adds 2D positional embeddings to BERT |
| **LAMBERT-base** | Transformer + Layout | Layout-Aware language Modeling using BERT, adds 2D positional features as attention bias |

All transformer models used their Base-sized variants. Results are averages over 3 runs.

### 2.5 Overall Baseline Results on Kleister Charity (Test Set, F1%)

| Model | Kleister Charity F1 (%) | Kleister NDA F1 (%) |
|-------|------------------------|---------------------|
| Flair | ~79.18 | -- |
| BERT-base | ~76.28 | -- |
| RoBERTa-base | ~81.53 | ~78.50 |
| LayoutLM-base | ~80.02 | -- |
| **LAMBERT-base** | **83.57** | **81.77** |

LAMBERT achieved the best results on both datasets, improving over the best text-only model (RoBERTa) by **2.04 F1 points** on Charity and **0.77 points** on NDA.

### 2.6 Per-Entity Difficulty Analysis

**Easier entities (higher F1):**
- `charity_number`: Appears in sequential text context; very high F1 scores across models. Unique format (numeric) makes it easier to identify.
- `report_date`: Also in sequential context; high F1. Well-defined format.

**Harder entities (lower F1):**
- `address__post_town`, `address__postcode`, `address__street_line`: Multiple addresses appear in documents (accountant, bank, charity itself); the system must identify specifically the charity's address.
- `charity_name`: Names vary in form (full names, acronyms, shortened versions).
- `income_annually_in_british_pounds`, `spending_annually_in_british_pounds`: Most challenging because:
  - Values often appear in tables (layout-dependent)
  - ~5% of cases require **scale inference** (e.g., values stated in thousands or millions, where the scale multiplier appears elsewhere in the document)
  - LAMBERT showed the biggest improvement on these: +4.03 on income, +5.60 on spending (compared to text-only models)

### 2.7 Additional Results

- **Autotagging baseline:** The autotagger's text-span-level F1 was inferior to almost all trained models at the document level, despite providing the training labels for those models.
- **Human performance:** Measured as inter-annotator agreement on 100 random documents. 4 annotators annotated 60 documents on text span level.
- **OCR comparison:** Multiple text extraction tools compared (pdf2djvu, Tesseract, Textract, combined). The best PDF processing tool was used for the reported baseline results.

### 2.8 What Makes This Dataset Challenging

1. **Long documents:** Average ~22 pages, far exceeding the context window of most transformer models (512 tokens). Documents must be chunked or truncated.
2. **Complex layouts:** Tables, multi-column formats, headers/footers, varying typography across 2,788 distinct report templates.
3. **Mixed document types:** Both scanned (requiring OCR) and born-digital PDFs (with embedded text layers of varying quality).
4. **Scale inference:** Monetary values may be stated in thousands or millions, requiring the model to understand context beyond the immediate value.
5. **Address disambiguation:** Multiple addresses in each document (charity, auditor, bank, etc.); only the charity address should be extracted.
6. **OCR errors:** No OCR method is perfect; errors propagate to downstream extraction.
7. **Decoy keys:** Some requested keys may have no corresponding value in the document.
8. **Performance gap:** Best baseline (83.57%) is far below simpler benchmarks like SROIE (98.17%), demonstrating the dataset's difficulty.
9. **Information not always explicitly stated:** Values may need to be inferred from context rather than directly read from text.

---

## 3. Related Benchmarks and Integration

### 3.1 DUE Benchmark (Document Understanding Evaluation)

Kleister Charity is included in the **DUE Benchmark** (NeurIPS 2021, Borchmann et al.).

- **Paper:** "DUE: End-to-End Document Understanding Benchmark"
- **Website:** https://duebenchmark.com/
- **Baselines:** T5-large and T5+2D (T5 with added horizontal/vertical positional bias from TILT)
- **Key challenge:** Encoder-decoder models cannot process entire Kleister Charity documents due to memory limitations, so only partial document content is consumed.
- **OCR engines supported:** microsoft_cv, tesseract, djvu
- **Evaluation output format:** JSONL with per-label F1, Precision, and Recall

### 3.2 Kleister Challenge (2021)

A formal challenge was organized on the Kleister datasets:
- **Website:** https://kleister.info/challenge/kleister-charity
- **GitHub organization:** https://github.com/kleister-challenge-2021
- **Repositories:** kleister-nda and kleister-charity

### 3.3 RealKIE Dataset (Related)

The **RealKIE Charities** dataset is based on the same UK Charity Commission document source but has a more extensive schema:
- 28 fields instead of 8
- Only 538 documents (vs. 2,778 in Kleister)
- Covers a wider mixture of data types including paragraph descriptions

---

## 4. Recent Work and Approaches (2022-2025)

### 4.1 Layout-Aware Models

The LayoutLM family continued to evolve after the original Kleister paper:

- **LayoutLMv2:** Achieved SOTA on Kleister-NDA (F1: 0.834 to 0.852). Incorporates visual features during pre-training.
- **LayoutLMv3:** Uses patch embeddings (ViT-style) instead of CNN backbone. Pre-trained with MLM, MIM, and word-patch alignment objectives. Achieves SOTA on multiple document AI benchmarks.
- **TILT (Text-Image-Layout Transformer):** Developed by Applica AI (same team as Kleister). End-to-end encoder-decoder model with layout attention bias and visual features. Evaluated primarily on DocVQA, CORD, SROIE.
- **LAMBERT:** The original best model on Kleister Charity. Published as a separate ICDAR 2021 paper.

### 4.2 OCR-Free Approaches

- **Donut (Document Understanding Transformer):** An OCR-free VDU model using a Swin Transformer encoder and BART decoder. Achieves SOTA on several benchmarks but specific Kleister Charity results not published.
- **Pix2Struct:** Pretrained by parsing masked web page screenshots to simplified HTML. Achieves SOTA on 6/9 tasks but is sensitive to hyperparameters.

### 4.3 LLM-Based Approaches

**Indico Data LLM Leaderboard (2024):**
- Tested multiple LLMs (Llama, Azure OpenAI GPT models, Google Gemini, AWS Bedrock models) on document extraction tasks including Kleister Charity.
- **Key finding:** Indico's own fine-tuned discriminative models (RoBERTa, DeBERTa) significantly outperformed all LLMs on structured extraction tasks.
- On NDA extraction: RoBERTa/DeBERTa achieved F1 > 0.8; best LLM (Claude V2) only reached F1 ~0.38.
- GPT-4 and Claude achieved perfect scores on document classification but struggled with extraction.
- LLMs excelled at summarization tasks.
- Conclusion: "A little prompt engineering goes a long way" but fine-tuned discriminative models remain superior for structured extraction.

### 4.4 Modern Document Processing Pipelines (2025)

- **spaCy + Docling:** Modular pipeline for PDF-to-structured-data conversion. Good for financial and organizational reports.
- **PDF-Extract-Kit:** Open-source toolkit for layout detection, formula recognition, and OCR from PDFs.
- **LlamaParse/LlamaIndex:** LLM-powered PDF parsing that understands document hierarchy and structure.
- **ACL 2025 paper on IE from visually rich documents:** LLM-based approach that organizes documents into independent textual segments for extraction.

### 4.5 Key Trends

1. **Specialized document AI models consistently outperform general-purpose LLMs** on structured extraction tasks like Kleister Charity.
2. **Layout information is critical** -- models incorporating 2D positional features (LAMBERT, LayoutLM, TILT, T5+2D) consistently outperform text-only baselines.
3. **Long document handling remains a bottleneck** -- most models truncate or chunk documents, losing context.
4. **LLMs offer zero-shot flexibility** but at high cost and inconsistent output for structured extraction.
5. **Hybrid approaches** (specialized models for extraction + LLMs for normalization/inference) may offer the best path forward.

---

## 5. Quick Reference

### Running Evaluation

```bash
# Download GEval
wget https://gonito.net/get/bin/geval
chmod u+x geval

# Generate your predictions as dev-0/out.tsv (same format as expected.tsv)
# Then evaluate:
./geval -t dev-0
```

### Expected Output Format

One line per document. Space-separated `key=value` pairs. Spaces and colons in values replaced with underscores:

```
charity_name=THE_EXAMPLE_FOUNDATION charity_number=1234567 report_date=2019-03-31 income_annually_in_british_pounds=500000.00 spending_annually_in_british_pounds=480000.00 address__post_town=MANCHESTER address__postcode=M1_1AA address__street_line=123_Example_Road
```

### Key Metrics Summary

| Metric | Value | Source |
|--------|-------|--------|
| Best baseline F1 (Kleister Charity) | 83.57% | LAMBERT-base, ICDAR 2021 paper |
| Worst baseline F1 (Kleister Charity) | ~76.28% | BERT-base, ICDAR 2021 paper |
| LAMBERT improvement over RoBERTa | +2.04 F1 | ICDAR 2021 paper |
| Number of documents | 2,788 | ICDAR 2021 paper |
| Number of pages | 61,643 | ICDAR 2021 paper |
| Number of entities | 21,612 | ICDAR 2021 paper |
| Number of entity types | 8 | GitHub repo |
| SROIE best (for comparison) | 98.17% | Reference benchmark |

---

## Sources

- **GitHub Repository:** https://github.com/applicaai/kleister-charity
- **ICDAR 2021 Paper:** https://arxiv.org/abs/2105.05796
- **Springer Publication:** https://link.springer.com/chapter/10.1007/978-3-030-86549-8_36
- **Earlier Paper Version:** https://arxiv.org/abs/2003.02356
- **DUE Benchmark:** https://duebenchmark.com/
- **DUE Baselines Code:** https://github.com/due-benchmark/baselines
- **Kleister Challenge:** https://kleister.info/challenge/kleister-charity
- **GEval Tool:** https://gitlab.com/filipg/geval
- **LAMBERT Paper:** https://link.springer.com/chapter/10.1007/978-3-030-86549-8_34
- **Applica AI Research:** https://www.applica.ai/about/research-papers
- **Indico LLM Leaderboard Analysis:** https://www.deep-analysis.net/atop-the-llm-leaderboard/
- **HuggingFace Document AI Blog:** https://huggingface.co/blog/document-ai
