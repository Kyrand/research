# Extracting Structured Facts from Scanned PDFs with LLMs: A Comprehensive Guide

A research survey of current best practices (2024-2026) for using large language models to extract structured, machine-readable facts from scanned PDF documents.

## Table of Contents

- [1. Text-Based Approaches](#1-text-based-approaches)
- [2. Multimodal / Vision Approaches](#2-multimodal--vision-approaches)
- [3. Hybrid Approaches](#3-hybrid-approaches)
- [4. Evaluation and Benchmarking](#4-evaluation-and-benchmarking)
- [5. Practical Hackathon Guide](#5-practical-hackathon-guide)
- [6. Recommended Architecture](#6-recommended-architecture)
- [Sources](#sources)

---

## 1. Text-Based Approaches

These approaches assume you have already extracted text from the PDF (via native text extraction or OCR) and want to use an LLM to parse structured facts from that text.

### 1.1 Prompting Strategies for Fact Extraction

#### Zero-Shot Prompting
The simplest approach: provide the LLM with text and a schema, and ask it to extract. Works surprisingly well for straightforward documents.

```
Extract the following fields from this document as JSON:
- patient_name (string)
- date_of_birth (string, YYYY-MM-DD format)
- diagnosis (string)
- medications (list of strings)

Document text:
{document_text}
```

Research findings: zero-shot with GPT-3.5 achieved 78.5% accuracy across 700,000+ sentences for clinical pharmacology extraction ([Meshkin et al., 2024](https://academic.oup.com/bib/article/25/5/bbae354/7739674)).

#### Few-Shot Prompting
Include 2-10 examples of input-output pairs before the actual extraction task. Research consistently shows this outperforms zero-shot for complex tasks:

- Claude models benefit *significantly* more from few-shot examples compared to GPT-4o variants ([Instructor blog](https://python.useinstructor.com/blog/2024/09/26/bad-schemas-could-break-your-llm-structured-outputs/))
- 10 well-chosen examples with Gemini-1.5 nearly matched a fine-tuned classifier's F1 (95% vs 96.2%) on claim matching ([Pisarevskaya & Zubiaga, 2025](https://arxiv.org/html/2501.10860v1))
- Heuristic prompts, alongside chain-of-thought, were the most effective across clinical NLP tasks ([PMC study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11036183/))

**Key insight**: Example *selection* matters enormously. Choose examples that cover edge cases and format variations, not just typical cases.

#### Chain-of-Thought for Document Understanding

Chain-of-thought (CoT) prompting asks the model to reason step-by-step before producing structured output. This is particularly valuable for documents with complex layouts or ambiguous fields.

The "Technical Specification Extraction Chain" pattern from [PLoP 2024](https://www.cs.wm.edu/~dcschmidt/PDF/Prompt_Patterns_for_Structured_Data_Extraction_from_Unstructured_Text___Final.pdf) combines three complementary patterns:
1. **Keyword Trigger Extractor** -- identify relevant sections
2. **Pattern Matcher** -- extract candidate values using patterns
3. **Semantic Extractor** -- validate and structure the final output

[Structured Chain-of-Thought (EMNLP 2024 Findings)](https://aclanthology.org/2024.findings-emnlp.948.pdf) specifically targets few-shot scenarios, combining structured output generation with step-by-step reasoning.

The [Focused CoT framework](https://web3.arxiv.org/pdf/2511.22176) shows that structuring *inputs* (not just outputs) leads to shorter reasoning traces and substantial token savings.

### 1.2 Structured Output Formats

#### Native JSON/Structured Output Modes

All major providers now support constrained structured output:

| Provider | Feature | Availability |
|----------|---------|-------------|
| OpenAI | `response_format: {type: "json_schema"}` | GPT-4o, GPT-4o-mini |
| Anthropic | Structured outputs (constrained decoding) | Claude Sonnet 4.5, Claude Opus 4.1 (beta) |
| Google | JSON mode with schema | Gemini 2.0 Flash, Gemini 2.5 Pro |
| OpenRouter | Pass-through structured outputs | Compatible models |

References: [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs), [Claude Structured Outputs](https://docs.claude.com/en/docs/build-with-claude/structured-outputs), [OpenRouter Structured Outputs](https://openrouter.ai/docs/guides/features/structured-outputs)

#### Function/Tool Calling for Extraction

Function calling repurposes the tool-use interface as a structured extraction mechanism. The LLM populates a function's argument schema without actually calling any function.

Key finding: Claude 3.5 Sonnet showed the most significant improvement with JSON tool calling, with F1 score increasing from 0.39 to 0.48. GPT-4o showed a more modest improvement from 0.40 to 0.42 ([Instructor research](https://python.useinstructor.com/blog/2024/09/26/bad-schemas-could-break-your-llm-structured-outputs/)).

#### The Instructor Library

[Instructor](https://python.useinstructor.com/) is the most popular Python library for structured LLM extraction (3M+ monthly downloads, 11k+ GitHub stars). It patches the OpenAI SDK to add a `response_model` parameter using Pydantic models.

```python
import instructor
from pydantic import BaseModel, field_validator
from typing import List, Optional

class InvoiceFact(BaseModel):
    vendor_name: str
    invoice_number: str
    date: str
    total_amount: float
    line_items: List[dict]

    @field_validator('date')
    def validate_date_format(cls, v):
        # Enforce YYYY-MM-DD format
        from datetime import datetime
        datetime.strptime(v, '%Y-%m-%d')
        return v

# Works with OpenRouter too
client = instructor.from_provider("openrouter/google/gemini-2.0-flash-001")
invoice = client.chat.completions.create(
    response_model=InvoiceFact,
    messages=[{"role": "user", "content": f"Extract invoice details:\n{ocr_text}"}],
    max_retries=3,  # Auto-retry on validation failure
)
```

Instructor supports 15+ providers including OpenAI, Anthropic, Google Gemini, Mistral, Ollama, DeepSeek, and OpenRouter ([Instructor OpenRouter integration](https://python.useinstructor.com/integrations/openrouter/)).

**Critical insight on schema design**: Field naming dramatically impacts performance. The Instructor team found that changing a single field name from `final_choice` to `answer` improved model accuracy from 4.5% to 95%. Schema design can matter more than model choice.

### 1.3 Chunking Strategies for Long Documents

Even with 1M+ token context windows (Gemini), chunking remains important because:
- Processing cost scales with input size
- "Needle in a haystack" degradation in long contexts
- Transformer processing time increases quadratically with context

**Recommended strategies** (from [NVIDIA 2024 benchmarks](https://developer.nvidia.com/blog/approaches-to-pdf-data-extraction-for-information-retrieval/), [Pinecone guide](https://www.pinecone.io/learn/chunking-strategies/)):

| Strategy | Best For | Details |
|----------|----------|---------|
| **Page-level** | Scanned PDFs | Highest accuracy in NVIDIA benchmarks; tables/figures stay together |
| **Recursive character splitting** | General text | Start here; respects paragraph/sentence boundaries |
| **Semantic chunking** | Multi-topic docs | Groups by meaning; higher quality but costs embedding calls |
| **Fixed-size with overlap** | Simple extraction | 256-512 tokens for factoid extraction; 1024+ for analytical |

For document fact extraction specifically, **page-level chunking is the default recommendation**. Each page becomes one extraction unit. This aligns naturally with the vision-based approach (one image per page).

Anthropic's [Contextual Retrieval](https://www.anthropic.com) (2024) offers an advanced technique: use a Claude instance to generate a contextual description for each chunk based on the full document, then prepend it to the chunk.

---

## 2. Multimodal / Vision Approaches

Vision approaches send PDF pages as images directly to a vision-language model (VLM), bypassing traditional OCR entirely.

### 2.1 Page-as-Image Extraction

The core idea: convert each PDF page to an image (PNG/JPEG at 300 DPI), send it to a vision model, and ask for structured extraction.

**Why this works well for scanned PDFs:**
- No OCR pipeline to build or maintain
- The VLM sees layout, formatting, tables, and text together
- Handles handwriting, stamps, and non-standard formatting
- Particularly strong on charts, infographics, and mixed content

**When OCR+text is better:**
- Very high-volume processing (vision tokens are more expensive)
- Documents with mostly clean, digital text
- When you need exact character-level fidelity
- OCR-based RAG generalizes better to unseen document types ([Lost in OCR Translation, 2025](https://arxiv.org/html/2505.05666v1))

### 2.2 PDF-to-Image Conversion Tools

#### PyMuPDF (Recommended)

```python
import pymupdf  # aka fitz

doc = pymupdf.open("scanned_document.pdf")
for page_num, page in enumerate(doc):
    # 300 DPI: zoom_factor = 300/72 = 4.17
    mat = pymupdf.Matrix(4.17, 4.17)
    pix = page.get_pixmap(matrix=mat)
    pix.save(f"page-{page_num}.png")
```

- No external dependencies
- Faster than pdf2image
- Matrix-based resolution control
- Also extracts embedded images via `page.get_images()`

Reference: [PyMuPDF documentation](https://pymupdf.readthedocs.io/en/latest/recipes-images.html), [Artifex guide](https://artifex.com/blog/converting-pdfs-to-images-with-pymupdf-a-complete-guide)

#### pdf2image

```python
from pdf2image import convert_from_path
images = convert_from_path("document.pdf", dpi=300)
for i, img in enumerate(images):
    img.save(f"page-{i}.png")
```

- Requires Poppler system dependency
- Slower for PNG (compression overhead)
- More widely used in tutorials

**Resolution guidance**: Default 96 DPI is insufficient for OCR-quality extraction. Use 300 DPI minimum. For small text or poor scans, consider 400-600 DPI.

### 2.3 Vision Models for Document Understanding

#### Proprietary Models (via API)

| Model | Strengths | Cost (per 1M tokens) | Via OpenRouter? |
|-------|-----------|----------------------|-----------------|
| **Gemini 2.0 Flash** | Best value; ~6000 pages/$1; near-perfect OCR | $0.10 in / $0.40 out | Yes |
| **GPT-4o** | Strong all-around; good at complex layouts | $2.50 in / $10.00 out | Yes |
| **GPT-4o-mini** | Best F1 on medical docs (55.6); very cost-effective | $0.15 in / $0.60 out | Yes |
| **Claude Sonnet** | Highest accuracy on industrial images; data privacy | $3.00 in / $15.00 out | Yes |
| **Gemini 2.5 Pro** | >95% OCR accuracy; 1M context window | $1.25 in / $10.00 out | Yes |

References: [OpenRouter pricing](https://openrouter.ai/pricing), [Vellum comparison](https://www.vellum.ai/blog/document-data-extraction-llms-vs-ocrs), [Koncile benchmark](https://www.koncile.ai/en/ressources/claude-gpt-or-gemini-which-is-the-best-llm-for-invoice-extraction)

#### Open-Source Vision Models

| Model | Size | Performance | Notes |
|-------|------|-------------|-------|
| **Qwen 2.5 VL** | 72B / 32B | ~75% accuracy (matches GPT-4o) | Best open-source for document extraction |
| **Qwen3-VL** | Various | OCR in 32 languages | Free on OpenRouter |
| **InternVL3** | 78B | Within 5-10% of proprietary | Strong document understanding |
| **Gemma3** | 27B | F1=41.3 on medical docs (best local) | Google's open model |
| **Molmo** | 1B/7B/72B | Matches GPT-4V, Claude 3.5 | Allen AI |

References: [Omni AI OCR benchmark](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr), [DataCamp VLM guide](https://www.datacamp.com/blog/top-vision-language-models), [BentoML guide](https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models)

#### Specialized OCR APIs

| Service | Cost per 1000 pages | Key Strengths |
|---------|---------------------|---------------|
| **Mistral OCR 3** | $1-2 | 88.9% handwriting, 96.6% tables, 2000 pages/min |
| **Gemini 2.0 Flash** | ~$1 | Best cost/accuracy ratio for bulk processing |
| **Azure Document Intelligence** | $1.50-10 | Strong table/form extraction |
| **AWS Textract** | $1.50-15 | AWS ecosystem integration |
| **Google Document AI** | $1.50-10 | Auto-tuned confidence thresholds |

References: [Mistral OCR announcement](https://mistral.ai/news/mistral-ocr), [Mistral OCR 3](https://mistral.ai/news/mistral-ocr-3), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

### 2.4 Zerox: Vision-First Document Extraction

[Zerox](https://github.com/getomni-ai/zerox) by OmniAI is an open-source library that converts PDFs to images and sends them to vision models for extraction. It started as a "weekend hack" that outperformed Unstructured/Textract at the same cost.

Key features:
- Python and Node.js SDKs
- Supports GPT-4o, Claude, Gemini, and others via LiteLLM
- **Structured extraction mode**: provide a JSON Schema and get structured data back
- Configurable: DPI, concurrency, orientation correction, separate extraction model
- `extractPerPage` option for page-level extraction

```python
from pyzerox import zerox
import asyncio

result = await zerox(
    file_path="document.pdf",
    model="openai/gpt-4o-mini",
    credentials={"apiKey": "your-key"},
    # Structured extraction
    extract_only=True,
    schema={
        "type": "object",
        "properties": {
            "vendor_name": {"type": "string"},
            "total": {"type": "number"},
            "date": {"type": "string"}
        }
    }
)
```

Reference: [Zerox documentation](https://docs.getomni.ai/zerox/overview), [Hacker News discussion](https://news.ycombinator.com/item?id=41048194)

### 2.5 Docling: IBM's Document Conversion Toolkit

[Docling](https://github.com/docling-project/docling) is IBM's open-source (MIT) toolkit for document conversion, with 10k+ GitHub stars in its first month (November 2024). It uses custom AI models for layout analysis and table extraction rather than traditional OCR.

Key capabilities:
- Supports PDF, DOCX, PPTX, XLSX, HTML, images, and more
- Custom LayoutLM-based layout analysis model
- TableFormer model for structured table extraction
- Exports to Markdown, HTML, JSON, DocTags
- Integrates with LangChain, LlamaIndex, CrewAI, Haystack
- Runs locally on a standard laptop
- Granite-Docling: 258M parameter model that rivals much larger systems

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")
markdown = result.document.export_to_markdown()
```

Reference: [Docling GitHub](https://github.com/docling-project/docling), [IBM Research blog](https://research.ibm.com/blog/docling-generative-AI), [Granite-Docling on HuggingFace](https://huggingface.co/ibm-granite/granite-docling-258M)

---

## 3. Hybrid Approaches

### 3.1 OCR + LLM Post-Processing

The traditional two-stage pipeline:

```
PDF --> OCR Engine --> Raw Text --> LLM --> Structured JSON
```

1. Extract text with a dedicated OCR tool (Tesseract, Azure, Mistral OCR)
2. Feed the text to an LLM with a Pydantic schema for structured extraction

This is often the most cost-effective approach: OCR is cheap (or free with Tesseract), and text-only LLM calls cost far less than vision calls.

A 2024 study on historical documents found that LLM post-correction of OCR output achieved approximately 1% character error rate -- effectively human-level transcription ([Ubicloud blog](https://www.ubicloud.com/blog/end-to-end-ocr-with-vision-language-models)).

### 3.2 Multi-Pass Extraction

**Coarse-then-fine strategy:**

1. **Pass 1 (Coarse)**: Use a fast/cheap model or OCR to extract raw text and identify document type
2. **Pass 2 (Fine)**: Use a more capable model with a specific schema for the identified document type
3. **Pass 3 (Validation)**: Optional verification pass with a different model or rule-based checks

Zerox supports this natively with its `extractionModel` parameter -- use one model for OCR and a different (potentially cheaper or more specialized) model for structured extraction.

**Document routing pattern**: Use a lightweight model to classify the document first, then route to type-specific extraction prompts. This avoids the cost of using a premium model with a generic prompt.

### 3.3 Ensemble / Cross-Model Voting

For high-stakes extraction (financial, medical, legal):

1. Run extraction on 2-3 different models (e.g., GPT-4o + Claude + Gemini)
2. Compare outputs field-by-field
3. Take majority vote, or flag disagreements for human review

No standardized framework exists for this, but it is a practical pattern emerging in production systems. Implementation is straightforward:

```python
from collections import Counter

def ensemble_extract(text, models, schema):
    results = [extract_with_model(text, model, schema) for model in models]
    # For each field, take majority vote
    final = {}
    for field in schema.model_fields:
        values = [getattr(r, field) for r in results]
        final[field] = Counter(values).most_common(1)[0][0]
    return final
```

### 3.4 Vision + OCR Text Combined

Send both the page image AND the OCR text to the LLM. This gives the model:
- Visual layout context from the image
- Precise text from OCR (which may be more accurate for clean text)

This is increasingly practical as multimodal models improve. The model can cross-reference visual positioning with text content.

---

## 4. Evaluation and Benchmarking

### 4.1 Computing F1 for Key-Value Extraction

The standard approach converts extraction results to key-value dictionaries and computes precision, recall, and F1:

```python
def compute_kv_f1(predicted: dict, ground_truth: dict, fuzzy=False, threshold=0.8):
    """
    Compute F1 score for key-value extraction.

    Args:
        predicted: dict of extracted key-value pairs
        ground_truth: dict of ground-truth key-value pairs
        fuzzy: if True, use Levenshtein similarity instead of exact match
        threshold: minimum similarity for fuzzy matching (0-1)
    """
    from Levenshtein import ratio as lev_ratio

    def matches(pred_val, gt_val):
        pred_str = str(pred_val).strip().lower()
        gt_str = str(gt_val).strip().lower()
        if fuzzy:
            return lev_ratio(pred_str, gt_str) >= threshold
        return pred_str == gt_str

    tp = 0  # true positives
    for key, gt_val in ground_truth.items():
        if key in predicted and matches(predicted[key], gt_val):
            tp += 1

    precision = tp / len(predicted) if predicted else 0
    recall = tp / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}
```

### 4.2 Evaluation Metrics Landscape

| Metric | Type | Best For |
|--------|------|----------|
| **Entity-level F1** | Exact match | Standard KV extraction |
| **ANLS** (Avg. Normalized Levenshtein Similarity) | Soft match | OCR-extracted text with minor errors |
| **KIEval** ([2025 paper](https://arxiv.org/html/2503.05488v2)) | Structured | Group-level and relational extraction |
| **Tree Edit Distance** | Soft match | Hierarchical/nested structures |
| **CER/WER** | Character/word level | Raw OCR quality |

**Recommendations:**
- Use **exact-match F1** as the primary metric for clean key-value extraction
- Add **ANLS** (threshold 0.5) when evaluating OCR-heavy pipelines to avoid penalizing minor transcription errors
- For nested structures (line items in invoices), consider **KIEval** or **TED**
- [John Snow Labs approach](https://www.johnsnowlabs.com/visual-document-understanding-benchmark-comparative-analysis-of-in-house-and-cloud-based-form-extraction-models/): convert forms to shallow JSON, compute precision/recall over keys+values with Levenshtein distance

### 4.3 Benchmark References

- **OCRBench v2**: open-source models scoring 75-83% ([olmOCR-Bench](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr))
- **Medical document benchmark**: GPT 4.1-mini F1=55.6, Gemma3 27B F1=41.3 on 1000 mock documents ([MedRxiv 2026](https://www.medrxiv.org/content/10.64898/2026.01.19.26344287v1.full.pdf))
- **Invoice extraction**: Mistral 63.75% reliability, 98.75% pure transcription ([Koncile](https://www.koncile.ai/en/ressources/mistral-ai-vs-chatgpt-reliable-solution-ocr-2025))
- **Google Document AI**: auto-computes optimal F1 threshold ([Google Cloud docs](https://docs.cloud.google.com/document-ai/docs/evaluate))

---

## 5. Practical Hackathon Guide

### 5.1 What Can Be Built in a Day

#### Option A: Vision-First Pipeline (2-3 hours to MVP)

The fastest path to a working demo. Minimal dependencies, maximum impact.

```
PDF --> PyMuPDF (page images) --> Vision LLM --> Structured JSON
```

Stack: PyMuPDF + Instructor + OpenRouter (or Gemini API directly)

#### Option B: OCR + Text Extraction Pipeline (half day)

More control, better for text-heavy documents.

```
PDF --> Docling/PyMuPDF4LLM --> Markdown text --> LLM + Instructor --> Structured JSON
```

Stack: Docling + Instructor + any LLM via OpenRouter

#### Option C: Zerox All-in-One (1-2 hours to MVP)

Zerox handles the entire pipeline. Just configure it and go.

```python
result = await zerox(
    file_path="document.pdf",
    model="openai/gpt-4o-mini",
    extract_only=True,
    schema=your_json_schema
)
```

#### Option D: Model Comparison Dashboard (full day)

Build a Streamlit/Gradio app that:
1. Takes a PDF upload
2. Runs extraction with 3-4 models
3. Displays results side-by-side
4. Computes F1 against ground truth

### 5.2 Best Models for OpenRouter Hackathon

#### Budget-Conscious (Best Bang-for-Buck)

| Model | Cost | Quality | Use Case |
|-------|------|---------|----------|
| **Gemini 2.0 Flash** | $0.10/$0.40 per 1M tokens | Excellent for vision+extraction | Default recommendation |
| **DeepSeek V3.2** | $0.25/$0.38 per 1M tokens | 93.5% field extraction accuracy | Text-only extraction |
| **Qwen3-VL** | Free (rate-limited) | Good OCR, 32 languages | Prototyping |
| **GPT-4o-mini** | $0.15/$0.60 per 1M tokens | Best F1 on medical docs | Balanced cost/quality |

#### Quality-First

| Model | Cost | When to Use |
|-------|------|-------------|
| **GPT-4o** | $2.50/$10.00 per 1M tokens | Complex layouts, highest reliability |
| **Claude Sonnet** | $3.00/$15.00 per 1M tokens | Best accuracy on industrial images; strong reasoning |
| **Gemini 2.5 Pro** | $1.25/$10.00 per 1M tokens | Very long documents (1M context) |

References: [OpenRouter pricing](https://openrouter.ai/pricing), [OpenRouter model comparison](https://compare-openrouter-models.pages.dev/), [InvertedStone calculator](https://invertedstone.com/calculators/openrouter-pricing)

#### Free Models for Prototyping

OpenRouter offers free-tier access (20 req/min, 200 req/day) to several capable models:
- Qwen3-VL (document AI, multilingual OCR)
- DeepSeek R1 (reasoning)
- Gemma 3 (general purpose)
- Nemotron Nano 2 VL (OCR-optimized)

Reference: [OpenRouter free models](https://openrouter.ai/collections/free-models)

### 5.3 Cost Estimates

For a hackathon processing ~100 scanned PDF pages:

| Approach | Estimated Cost |
|----------|---------------|
| Gemini 2.0 Flash (vision) | ~$0.02 |
| GPT-4o-mini (vision) | ~$0.10-0.50 |
| Mistral OCR 3 + text LLM | ~$0.20-0.50 |
| GPT-4o (vision) | ~$2-5 |
| Claude Sonnet (vision) | ~$1.50-3 |
| Free OpenRouter models | $0 (within rate limits) |

For 1,000 pages, multiply by roughly 10x. The key insight: with Gemini 2.0 Flash or free OpenRouter models, cost is essentially a non-issue for hackathon-scale work.

### 5.4 Minimal Working Example

Here is a complete, minimal pipeline you could get running in under an hour:

```python
"""
Minimal PDF fact extraction pipeline.
Requirements: pip install pymupdf instructor openai pillow
"""
import pymupdf
import instructor
import base64
from pydantic import BaseModel
from typing import List, Optional

# 1. Define your extraction schema
class ExtractedFacts(BaseModel):
    document_type: str
    date: Optional[str] = None
    key_entities: List[str] = []
    amounts: List[dict] = []      # [{"description": ..., "value": ...}]
    summary: str

# 2. Convert PDF pages to base64 images
def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    doc = pymupdf.open(pdf_path)
    images = []
    zoom = dpi / 72
    mat = pymupdf.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()
        images.append(b64)
    return images

# 3. Extract with a vision model via OpenRouter
client = instructor.from_provider("openrouter/google/gemini-2.0-flash-001")

def extract_facts(pdf_path: str) -> ExtractedFacts:
    images = pdf_to_images(pdf_path)

    # Build message with all page images
    content = [{"type": "text", "text": "Extract all key facts from this document."}]
    for b64_img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_img}"}
        })

    return client.chat.completions.create(
        response_model=ExtractedFacts,
        messages=[{"role": "user", "content": content}],
        max_retries=2,
    )

# Usage
facts = extract_facts("scanned_invoice.pdf")
print(facts.model_dump_json(indent=2))
```

### 5.5 Prompting Tips for Best Results

1. **Be explicit about output format**: "Return dates in YYYY-MM-DD format. Return amounts as floats without currency symbols."
2. **Name fields intuitively**: Use `patient_name` not `entity_1`. Field naming affects accuracy more than you'd expect.
3. **Make fields optional when appropriate**: Use `Optional[str] = None` for fields that may not be present in every document.
4. **Provide extraction instructions in the prompt**: "If a field is not visible in the document, return null rather than guessing."
5. **Use XML-style prompting for Claude**: Claude was trained on XML-structured inputs and responds well to `<document>`, `<instructions>`, `<schema>` tags.
6. **For few-shot: show the edge cases**: Include examples with missing fields, unusual formatting, or ambiguous values.

---

## 6. Recommended Architecture

### For a Hackathon (Speed + Simplicity)

```
                    +------------------+
                    |   PDF Document   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  PyMuPDF: pages   |
                    |  to PNG at 300dpi |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Vision LLM via   |
                    | OpenRouter +     |
                    | Instructor       |
                    | (Gemini Flash or |
                    |  GPT-4o-mini)    |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Pydantic Model   |
                    | (validated JSON) |
                    +------------------+
```

### For Production (Accuracy + Cost Efficiency)

```
    PDF Document
         |
    +----v----+
    | Classify |  <-- Lightweight model: identify doc type
    +----+----+
         |
    +----v-----------+     +----------------+
    | Route to       |---->| Scanned/Image? |
    | appropriate    |     | Use Vision LLM |
    | pipeline       |     +----------------+
    |                |
    |                |---->| Digital text?  |
    |                |     | Use OCR/text   |
    +----------------+     | extraction +   |
                           | text-only LLM  |
                           +-------+--------+
                                   |
                           +-------v--------+
                           | Validate with  |
                           | Pydantic +     |
                           | business rules |
                           +-------+--------+
                                   |
                           +-------v--------+
                           | (Optional)     |
                           | Second model   |
                           | verification   |
                           +----------------+
```

---

## Sources

### Key Tools and Libraries

- [Instructor](https://python.useinstructor.com/) -- Structured LLM outputs with Pydantic validation
- [Zerox (OmniAI)](https://github.com/getomni-ai/zerox) -- Vision-based OCR and document extraction
- [Docling (IBM)](https://github.com/docling-project/docling) -- AI-driven document conversion toolkit
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) -- PDF text/image extraction and conversion
- [PyMuPDF4LLM](https://medium.com/@danushidk507/using-pymupdf4llm-a-practical-guide-for-pdf-extraction-in-llm-rag-environments-63649915abbf) -- PDF to Markdown for LLM consumption
- [OpenRouter](https://openrouter.ai/) -- Unified API for 290+ LLM models
- [LLMWhisperer](https://unstract.com/) -- Layout-preserving text extraction

### Blog Posts and Tutorials

- [Unstract: LLMs for Structured Data Extraction from PDFs](https://unstract.com/blog/comparing-approaches-for-using-llms-for-structured-data-extraction-from-pdfs/) -- Comprehensive comparison of approaches
- [Vellum: Document Data Extraction - LLMs vs OCRs](https://www.vellum.ai/blog/document-data-extraction-llms-vs-ocrs) -- Cost and accuracy trade-offs
- [NVIDIA: Approaches to PDF Data Extraction](https://developer.nvidia.com/blog/approaches-to-pdf-data-extraction-for-information-retrieval/) -- Page-level chunking benchmarks
- [LlamaIndex: Beyond OCR - How LLMs Are Revolutionizing PDF Parsing](https://www.llamaindex.ai/blog/beyond-ocr-how-llms-are-revolutionizing-pdf-parsing) -- Vision-language model approaches
- [6000 Pages per Dollar: Gemini 2.0 Flash](https://medium.com/ai-simplified-in-plain-english/6-000-pages-per-dollar-how-gemini-2-0-flash-crushes-pdf-processing-costs-19637618243a) -- Cost analysis
- [Ingesting Millions of PDFs with Gemini 2.0](https://www.sergey.fyi/articles/gemini-flash-2) -- Scale considerations
- [Instructor Blog: Bad Schemas Could Break Your LLM Structured Outputs](https://python.useinstructor.com/blog/2024/09/26/bad-schemas-could-break-your-llm-structured-outputs/) -- Schema design impact
- [Structured Data Extraction with Instructor and LLMs](https://learnbybuilding.ai/tutorial/structured-data-extraction-with-instructor-and-llms/) -- Step-by-step tutorial
- [Pydantic + LLMs Guide](https://pydantic.dev/articles/llm-intro) -- Schema definition best practices
- [Agenta: The Guide to Structured Outputs and Function Calling](https://agenta.ai/blog/the-guide-to-structured-outputs-and-function-calling-with-llms) -- Comprehensive comparison
- [Mindee: LLMs vs OCR APIs -- The Hidden Cost Trap](https://www.mindee.com/blog/llm-vs-ocr-api-cost-comparison) -- Cost analysis for production

### Research Papers

- [KIEval: Evaluation Metric for Document Key Information Extraction (2025)](https://arxiv.org/html/2503.05488v2) -- New evaluation metric
- [Structured Chain-of-Thought Prompting (EMNLP 2024 Findings)](https://aclanthology.org/2024.findings-emnlp.948.pdf) -- CoT for structured output
- [Prompt Patterns for Structured Data Extraction (PLoP 2024)](https://www.cs.wm.edu/~dcschmidt/PDF/Prompt_Patterns_for_Structured_Data_Extraction_from_Unstructured_Text___Final.pdf) -- Systematic prompt engineering
- [Benchmarking LLM-based IE Tools for Medical Documents (2026)](https://www.medrxiv.org/content/10.64898/2026.01.19.26344287v1.full.pdf) -- F1 benchmarks across models
- [ColPali: Efficient Document Retrieval with Vision Language Models](https://huggingface.co/blog/manu/colpali) -- Vision-based document retrieval
- [Lost in OCR Translation? Vision-Based Document Retrieval (2025)](https://arxiv.org/html/2505.05666v1) -- OCR vs vision for RAG
- [Zero-shot and Few-shot Learning for Claim Matching (2025)](https://arxiv.org/html/2501.10860v1) -- Few-shot effectiveness

### Model and Pricing References

- [OpenRouter Pricing](https://openrouter.ai/pricing) -- Current per-token pricing
- [OpenRouter Free Models](https://openrouter.ai/collections/free-models) -- Free tier options
- [OpenRouter Model Comparison](https://compare-openrouter-models.pages.dev/) -- Side-by-side comparison tool
- [Gemini API Pricing](https://ai.google.dev/gemini-api/docs/pricing) -- Google Gemini pricing
- [Mistral OCR Announcement](https://mistral.ai/news/mistral-ocr) -- Mistral OCR capabilities
- [Mistral OCR 3](https://mistral.ai/news/mistral-ocr-3) -- Latest OCR model
- [DeepSeek OCR vs Mistral OCR Benchmark](https://sparkco.ai/blog/deepseek-ocr-vs-mistral-ocr-benchmark-analysis-2025) -- Comparison

### Benchmarks and Evaluations

- [Omni AI: Benchmarking Open-Source OCR Models](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr) -- 1000-document benchmark
- [John Snow Labs: Visual Document Understanding Benchmark](https://www.johnsnowlabs.com/visual-document-understanding-benchmark-comparative-analysis-of-in-house-and-cloud-based-form-extraction-models/) -- F1 methodology
- [Google Document AI Evaluation](https://docs.cloud.google.com/document-ai/docs/evaluate) -- Evaluation methodology
- [Koncile: Claude vs GPT vs Gemini for Invoice Extraction](https://www.koncile.ai/en/ressources/claude-gpt-or-gemini-which-is-the-best-llm-for-invoice-extraction) -- Practical comparison
- [Algodocs: Best LLM Models for Document Processing 2025](https://algodocs.com/best-llm-models-for-document-processing-in-2025/) -- Model comparison
