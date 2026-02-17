# Kleister Charity Playgroup Preparation

Suggested approaches for a playgroup session tackling fact extraction from UK charity financial reports using the [Kleister Charity dataset](https://github.com/applicaai/kleister-charity) ([paper](https://arxiv.org/abs/2105.05796)).

## The Dataset at a Glance

**What it is:** 2,788 scanned/printed UK Charity Commission annual reports (61,643 pages total). Each document averages ~22 pages. The task is to extract 8 structured fields per document.

**The 8 fields to extract:**

| Field | Format | Difficulty |
|-------|--------|-----------|
| `charity_number` | Digits | Easy — distinctive format |
| `report_date` | `YYYY-MM-DD` | Easy — pattern-matchable |
| `charity_name` | UPPERCASE | Medium — acronyms, variations |
| `address__post_town` | UPPERCASE | Medium — multiple addresses per doc |
| `address__postcode` | UPPERCASE | Medium — must pick the right address |
| `address__street_line` | Mixed case | Medium — formatting varies |
| `income_annually_in_british_pounds` | `1000000.00` | Hard — scale inference needed |
| `spending_annually_in_british_pounds` | `1000000.00` | Hard — scale inference needed |

**What makes it hard:**
- Documents are long (up to 20 pages in the subset we'll use)
- Mix of scanned and born-digital PDFs — noisy OCR
- Multiple addresses appear in each doc (charity address, auditor, bank, etc.) — must identify the right one
- ~5% of monetary values require scale inference ("expressed in thousands")
- Some fields are "decoy keys" — no value should be returned
- Best published ML result is only 83.57% F1 (LAMBERT, ICDAR 2021)

**What's provided with the dataset:**
- Pre-extracted text from 3 OCR engines: pdf2djvu, Tesseract 4.1, Amazon Textract
- A "best" combined text column
- Gold standard `key=value` pairs for evaluation
- The raw PDFs themselves

**Evaluation:** F1 score on upper-cased extracted values, with per-entity breakdowns.

## Suggested Approaches

### Approach 1: Text-Only Extraction with Structured Output (Start Here)

**Time:** 1-2 hours to baseline | **Cost:** Minimal | **Complexity:** Low

Use the pre-extracted OCR text (already provided in the dataset) and prompt an LLM to extract the 8 fields as structured JSON.

**Why start here:**
- No image processing needed — uses text already in the dataset
- Fast iteration cycle — change prompts and re-run
- Establishes a baseline to compare other approaches against

**Stack:** Python + Instructor + OpenRouter

```python
import instructor
from pydantic import BaseModel, Field
from typing import Optional

class CharityExtraction(BaseModel):
    """Extracted facts from a UK charity annual report."""
    charity_name: Optional[str] = Field(None, description="Full charity name in UPPERCASE")
    charity_number: Optional[str] = Field(None, description="Registered charity number")
    report_date: Optional[str] = Field(None, description="Report date in YYYY-MM-DD format")
    address_street_line: Optional[str] = Field(None, description="Street address of the charity")
    address_post_town: Optional[str] = Field(None, description="Town/city in UPPERCASE")
    address_postcode: Optional[str] = Field(None, description="UK postcode in UPPERCASE")
    income_annually_in_british_pounds: Optional[str] = Field(
        None, description="Total annual income as a number with 2 decimal places, e.g. 1000000.00. "
        "If figures are stated 'in thousands', multiply by 1000."
    )
    spending_annually_in_british_pounds: Optional[str] = Field(
        None, description="Total annual spending/expenditure as a number with 2 decimal places. "
        "If figures are stated 'in thousands', multiply by 1000."
    )

client = instructor.from_provider("openrouter/google/gemini-2.0-flash-001")

def extract_from_text(ocr_text: str) -> CharityExtraction:
    return client.chat.completions.create(
        response_model=CharityExtraction,
        messages=[{
            "role": "user",
            "content": f"""Extract the specified facts from this UK charity annual report.

IMPORTANT RULES:
- For income/spending: check if figures are "in thousands" or "in millions" and scale accordingly
- For addresses: extract the CHARITY's registered address, not the auditor's or bank's
- If a field is not present in the document, return null
- Charity name should be in UPPERCASE
- Report date should be the period end date in YYYY-MM-DD format
- Monetary values should have exactly 2 decimal places

DOCUMENT TEXT:
{ocr_text}"""
        }],
        max_retries=2,
    )
```

**Experiments to run:**
1. Zero-shot (prompt only, no examples) — establish baseline
2. Few-shot (add 3-5 gold standard examples to the prompt)
3. Compare OCR sources: `text_best` vs `text_tesseract` vs `text_textract`
4. Compare models: Gemini 2.0 Flash vs GPT-4o-mini vs Claude Sonnet

**Key insight from research:** Field naming in Pydantic models dramatically affects accuracy. Using descriptive names like `income_annually_in_british_pounds` (matching the dataset field names) is better than generic names like `field_7`.

---

### Approach 2: Vision-Based Extraction (Page Images)

**Time:** 2-3 hours | **Cost:** Low-moderate | **Complexity:** Medium

Convert PDF pages to images and send them directly to a vision-language model. The model sees the layout, tables, and formatting naturally — no OCR pipeline needed.

**Why try this:**
- Layout is critical for this dataset — income/spending often appear in tables
- LAMBERT's biggest gains over text-only models were on income (+4.03 F1) and spending (+5.60 F1), both table-dependent fields
- Vision models handle noisy scanned documents better than OCR pipelines
- Simpler pipeline (no OCR quality issues)

**Stack:** PyMuPDF + Instructor + OpenRouter

```python
import pymupdf
import base64
import instructor
from typing import Optional

def pdf_pages_to_base64(pdf_path: str, dpi: int = 300) -> list[str]:
    """Convert PDF pages to base64-encoded PNG images."""
    doc = pymupdf.open(pdf_path)
    images = []
    zoom = dpi / 72
    mat = pymupdf.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(base64.b64encode(pix.tobytes("png")).decode())
    return images

def extract_from_pdf(pdf_path: str) -> CharityExtraction:
    page_images = pdf_pages_to_base64(pdf_path)

    content = [{
        "type": "text",
        "text": """Extract facts from this UK charity annual report.
Look for: charity name, number, report date, registered address, annual income, annual spending.
For income/spending: if figures are 'in thousands', multiply by 1000. Return 2 decimal places.
For address: extract the CHARITY's address, not the auditor's or bank's."""
    }]
    for b64 in page_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    client = instructor.from_provider("openrouter/google/gemini-2.0-flash-001")
    return client.chat.completions.create(
        response_model=CharityExtraction,
        messages=[{"role": "user", "content": content}],
        max_retries=2,
    )
```

**Experiments to run:**
1. All pages vs first 3-5 pages only (many fields appear early in the document)
2. DPI comparison: 200 vs 300 vs 400 (trade-off between quality and token cost)
3. Compare: Gemini 2.0 Flash (cheapest vision) vs GPT-4o-mini vs Claude Sonnet

**Cost note:** At 300 DPI, a 20-page document is roughly 20 images. With Gemini 2.0 Flash ($0.10/1M input tokens), this costs fractions of a cent per document.

---

### Approach 3: Hybrid — OCR Text + Vision on Key Pages

**Time:** 3-4 hours | **Cost:** Low | **Complexity:** Medium-high

Use OCR text for "easy" fields (charity number, name, date, address) and vision on specific pages for "hard" fields (income, spending in tables).

**Why try this:**
- Fastest/cheapest for easy fields — text extraction is nearly free
- Reserves expensive vision calls for where they add value (tables, layout-dependent data)
- Can target specific pages rather than sending the full document

**Strategy:**

```
Pass 1 (Text): Extract easy fields from OCR text
  → charity_number, charity_name, report_date, address fields

Pass 2 (Vision): Send financial statement pages to vision model
  → income_annually_in_british_pounds, spending_annually_in_british_pounds

Pass 3 (Optional): Validation pass — check extracted values for consistency
  → Does postcode match post_town? Is income > spending (usually)?
```

**Implementation sketch:**

```python
# Pass 1: cheap text extraction for straightforward fields
easy_fields = extract_easy_fields_from_text(ocr_text)  # text-only LLM call

# Pass 2: find financial pages and use vision
financial_pages = identify_financial_pages(ocr_text)  # keyword search for "income", "expenditure"
hard_fields = extract_financials_from_images(pdf_path, financial_pages)

# Merge
result = {**easy_fields.model_dump(), **hard_fields.model_dump()}
```

---

### Approach 4: Multi-Model Ensemble / Comparison

**Time:** Full session | **Cost:** Moderate | **Complexity:** High

Run extraction with 3+ different models and compare their outputs — majority voting, disagreement analysis, or best-of-N selection.

**Why try this:**
- Exposes model-specific strengths and weaknesses
- Majority voting often outperforms any single model
- Makes for interesting playgroup discussion — which models are better at what?

**Strategy:**

```python
models = [
    "openrouter/google/gemini-2.0-flash-001",
    "openrouter/openai/gpt-4o-mini",
    "openrouter/anthropic/claude-3.5-sonnet",
]

results = {}
for model in models:
    client = instructor.from_provider(model)
    results[model] = extract(client, document_text)

# Compare per-field agreement
for field in CharityExtraction.model_fields:
    values = {m: getattr(results[m], field) for m in models}
    if len(set(v for v in values.values() if v)) > 1:
        print(f"DISAGREEMENT on {field}: {values}")
```

**Experiments:**
1. Per-model F1 breakdown by entity type
2. Majority voting F1 vs best single model
3. Cost-per-F1-point analysis

---

## Practical Recommendations for the Playgroup

### Suggested session structure

| Time | Activity |
|------|----------|
| First 30 min | Setup: load dataset, parse TSV, write evaluation function |
| Next 2 hours | **Approach 1** — text-only baseline with zero-shot, then few-shot |
| Next 1.5 hours | **Approach 2** — vision-based extraction on a subset |
| Next 1 hour | Compare approaches, per-entity analysis, discuss findings |
| Remaining time | Try Approach 3 or 4, or deep-dive on hardest entities |

### Model recommendations via OpenRouter

| Model | Best For | Cost |
|-------|----------|------|
| **Gemini 2.0 Flash** | Default — best value for both text and vision | $0.10/$0.40 per 1M tokens |
| **GPT-4o-mini** | Strong alternative, good structured output | $0.15/$0.60 per 1M tokens |
| **Claude Sonnet** | Highest accuracy, best reasoning | $3.00/$15.00 per 1M tokens |
| **Qwen3-VL** | Free tier for prototyping | Free (rate-limited) |
| **DeepSeek V3** | Cheapest paid option for text-only | $0.25/$0.38 per 1M tokens |

### Evaluation helper

```python
def evaluate(predicted: dict, gold: dict) -> dict:
    """Compute F1 for a single document extraction."""
    # Gold format: "key1=val1 key2=val2 ..."
    # Both predicted and gold values should be uppercased for comparison

    tp = fp = fn = 0
    for key, gold_val in gold.items():
        pred_val = predicted.get(key)
        if pred_val is None:
            fn += 1
        elif pred_val.upper().replace(" ", "_") == gold_val.upper().replace(" ", "_"):
            tp += 1
        else:
            fp += 1
            fn += 1  # wrong value counts as both FP and FN

    # Keys predicted but not in gold
    for key in predicted:
        if key not in gold and predicted[key] is not None:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
```

### Things likely to trip people up

1. **Scale inference** — "The figures are expressed in thousands of pounds." Income of "1,234" actually means "1234000.00". This is the single hardest aspect.
2. **Multiple addresses** — Each report has the charity's address, the auditor's address, the bank's address, sometimes a second office. The prompt must be very specific about wanting the charity's *registered* address.
3. **Decoy keys** — Some documents list keys to extract but the value genuinely isn't in the document. The model needs to return `null`, not hallucinate.
4. **Date formatting** — Reports cover a period (e.g., "1 April 2019 to 31 March 2020"). The `report_date` is the *end* date: `2020-03-31`.
5. **Gold standard format** — Values have spaces replaced with underscores. Make sure your evaluation normalises this.
6. **OCR noise** — Tesseract/djvu text can be garbled. Compare `text_best` vs the individual OCR outputs to see which works best for each document.

### What to install

```bash
pip install instructor pymupdf pydantic openai python-Levenshtein
```

## Reference: Published Baselines

The best published result on this dataset is **83.57% F1** using LAMBERT (a layout-aware BERT variant). This required:
- Custom autotagging pipeline to create training labels
- Fine-tuned sequence labelling model with 2D positional features
- Post-processing for text normalisation and value selection

A playgroup using LLMs with zero/few-shot prompting is unlikely to beat this, but can plausibly reach 60-80% F1 depending on the approach. The interesting question is: which fields do LLMs handle well (likely charity_number, report_date, charity_name) and which do they struggle with (income/spending scale inference, address disambiguation)?

## Sources

- [Kleister Charity Dataset](https://github.com/applicaai/kleister-charity) — GitHub repo with data and evaluation
- [Kleister Paper (ICDAR 2021)](https://arxiv.org/abs/2105.05796) — Dataset description and baselines
- [Instructor Library](https://python.useinstructor.com/) — Structured LLM outputs with Pydantic
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF to image conversion
- [OpenRouter](https://openrouter.ai/) — Unified API for LLM models
- [Zerox](https://github.com/getomni-ai/zerox) — Vision-based document extraction
- [Docling](https://github.com/docling-project/docling) — IBM's document conversion toolkit
- [NVIDIA: PDF Data Extraction Approaches](https://developer.nvidia.com/blog/approaches-to-pdf-data-extraction-for-information-retrieval/) — Chunking benchmarks
- [Instructor Blog: Schema Design Impact](https://python.useinstructor.com/blog/2024/09/26/bad-schemas-could-break-your-llm-structured-outputs/) — Field naming matters
