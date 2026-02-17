# LLM PDF Fact Extraction - Research Notes

## 2026-02-17 - Starting research

Investigating best practices for using LLMs to extract structured facts from scanned PDF documents. Covering text-based, vision-based, and hybrid approaches, plus evaluation methods and practical hackathon considerations.

---

## Search queries performed

1. "LLM structured data extraction from PDF documents best practices 2024 2025"
2. "GPT-4 Claude structured output JSON extraction from documents prompting strategies 2024"
3. "vision LLM PDF page image extraction GPT-4V Claude 3 Gemini document understanding 2024 2025"
4. "PDF to images Python pdf2image PyMuPDF document AI extraction pipeline 2024"
5. "few-shot vs zero-shot prompting LLM information extraction comparison 2024 2025"
6. "chain of thought prompting document understanding extraction structured data 2024"
7. "F1 score key-value extraction evaluation benchmark document AI 2024 2025"
8. "OpenRouter API cheapest best models document extraction cost comparison 2024 2025"
9. "instructor library pydantic LLM structured extraction tutorial Python 2024 2025"
10. "zerox getomni-ai PDF vision OCR extraction tutorial usage 2024"
11. "hybrid OCR vision LLM multi-pass extraction ensemble voting document 2024 2025"
12. "LLM chunking strategy long documents extraction token limits 2024 2025"
13. "docling IBM document extraction Python library 2024 2025"
14. "LLM document extraction hackathon build in one day tutorial quickstart 2024 2025"
15. "mistral OCR API document extraction 2025 benchmark"
16. "LLM function calling tool use extract structured data from document example code 2024"
17. "Gemini 2.0 Flash document extraction PDF cost per page 2025"

---

## Key findings - Text-based approaches

### Structured output modes
- OpenAI: `response_format: {type: "json_schema", ...}` with Pydantic schemas
- Claude: Structured outputs public beta for Claude Sonnet 4.5 and Opus 4.1 via constrained decoding
- OpenRouter: Supports structured outputs for compatible models, single API key across 290+ models
- Instructor library (3M+ monthly downloads): patches OpenAI SDK to add `response_model` parameter, works with 15+ providers including OpenRouter

### Field naming matters dramatically
- Research from Instructor team: changing a single field name from `final_choice` to `answer` improved accuracy from 4.5% to 95%
- Schema design can have more impact than model selection

### Few-shot vs zero-shot
- Claude models benefit significantly more from few-shot examples than GPT-4o variants
- 10 well-chosen examples with Gemini-1.5 nearly matched fine-tuned classifier F1 (95% vs 96.2%)
- For complex tasks, few-shot consistently outperforms zero-shot
- Task-specific prompt tailoring is vital

### Chain-of-thought for extraction
- PLoP 2024: "Technical Specification Extraction Chain" combines Keyword Trigger Extractor -> Pattern Matcher -> Semantic Extractor
- Structured CoT (EMNLP 2024): combines structured output with step-by-step reasoning
- Focused CoT (F-CoT): structuring inputs leads to shorter reasoning traces and token savings

### Chunking strategies for long documents
- Page-level chunking achieved highest accuracy in NVIDIA 2024 benchmarks
- Factoid queries: 256-512 tokens optimal; analytical queries: 1024+ tokens
- Recursive character splitting recommended as starting point
- Anthropic's contextual retrieval (2024): prepend chunk context from full document

---

## Key findings - Vision/multimodal approaches

### Direct page-as-image approach
- Zerox library: converts PDF pages to images, sends to vision LLM, gets markdown back
- Originally a "weekend hack" that outperformed Unstructured/Textract at same cost
- Supports GPT-4o, Claude 3, Gemini via LiteLLM
- Also supports structured extraction with JSON Schema

### Vision model benchmarks
- Qwen 2.5 VL (72B and 32B): ~75% accuracy on 1000-doc JSON extraction benchmark (matching GPT-4o)
- Both edged out mistral-ocr (72.2%)
- On medical documents: GPT 4.1-mini best at F1=55.6; Gemma3 27B best local at F1=41.3
- Claude 3 and Gemini scored highest median accuracy on industrial images
- Open-source models now within 5-10% of proprietary

### PDF-to-image conversion tools
- PyMuPDF: faster, no external deps, Matrix zoom for resolution control
- pdf2image: requires Poppler, slower for PNG but widely used
- Zerox: uses GraphicsMagick + Ghostscript internally
- Default 96 DPI not enough for OCR; use 300 DPI minimum

### Specialized OCR APIs
- Mistral OCR 3: $1-2 per 1000 pages, 88.9% handwriting accuracy, 96.6% tables
- Gemini 2.0 Flash: ~$1 per 1000 pages (~6000 pages per dollar), near-perfect OCR accuracy
- Both significantly cheaper than traditional cloud OCR services

---

## Key findings - Hybrid approaches

### OCR + LLM post-processing
- Traditional pipeline: OCR -> clean text -> LLM extraction
- LLM post-correction of OCR output achieved ~1% character error rate on historical docs
- Hybrid systems: LLM falls back to strict OCR reading for critical portions

### Multi-pass extraction
- Not a formally named technique, but widely practiced:
  - Pass 1: OCR or vision model for text extraction
  - Pass 2: LLM for structured fact extraction from text
  - Pass 3 (optional): Validation/correction pass
- Zerox supports separate OCR and extraction models via `extractionModel` parameter

### Ensemble/voting
- Using multiple models and comparing outputs emerging for high-stakes processing
- No standardized framework found, but practical pattern
- Cross-model validation: run extraction on 2-3 models, take consensus

---

## Key findings - Evaluation

### Standard metrics
- Entity-level F1: most common, exact-match based
- ANLS (Average Normalized Levenshtein Similarity): soft matching for OCR errors
- KIEval (2025): addresses F1 limitations for group-level extraction
- Tree Edit Distance (TED): soft-match to avoid over-penalization

### Computing F1 for key-value extraction
- Convert predicted and ground-truth to flat key-value dictionaries
- Exact match or fuzzy match (Levenshtein distance) per field
- Precision = correct extractions / total predicted
- Recall = correct extractions / total ground truth
- F1 = 2 * (precision * recall) / (precision + recall)
- John Snow Labs approach: convert forms to shallow JSON, compute over keys+values

### Benchmark references
- OCRBench v2: open-source models scoring 75-83%
- olmOCR-Bench leaderboard for document parsing
- Google Document AI auto-computes optimal F1 threshold

---

## Key findings - Practical hackathon considerations

### What can be built in a day
1. PDF -> vision LLM -> structured JSON pipeline (simplest, 2-3 hours)
2. PDF -> OCR -> LLM extraction with Pydantic validation (half day)
3. Side-by-side model comparison dashboard (full day)
4. Multi-document batch extraction with evaluation (full day)

### Best bang-for-buck with OpenRouter
- Free tier models: Qwen3-VL, DeepSeek R1, Gemma 3 (20 req/min, 200 req/day)
- Budget paid: DeepSeek V3.2 at $0.25/$0.38 per 1M tokens
- Mid-tier: Gemini 2.0 Flash at $0.10/$0.40 per 1M tokens (best value for vision)
- Premium: GPT-4o, Claude Sonnet (for highest accuracy needs)

### Key libraries for quick prototyping
- **Instructor** (Python): structured extraction with any LLM, Pydantic validation, auto-retries
- **Zerox** (Python/Node): vision-based OCR+extraction, supports schema-based extraction
- **Docling** (Python): IBM's document conversion toolkit, layout analysis, table extraction
- **PyMuPDF4LLM**: PDF to Markdown for LLM consumption
- **LangChain**: broader framework with PDF loaders and extraction chains

### Cost estimates (per 1000 pages)
- Gemini 2.0 Flash: ~$1
- Mistral OCR 3: $1-2
- GPT-4o mini (vision): ~$5-10
- GPT-4o (vision): ~$20-50
- Claude Sonnet (vision): ~$15-30
- Traditional OCR (Azure/AWS): $1.50-15

---

## Key sources found

### Blog posts and tutorials
- Unstract: "LLMs for Structured Data Extraction from PDFs" (2026)
- Vellum: "Document Data Extraction in 2026: LLMs vs OCRs"
- Phil Schmid: "Structured Outputs from PDFs with Gemini 2.0"
- NVIDIA Technical Blog: "Approaches to PDF Data Extraction"
- Medium: "6,000 Pages per Dollar: Gemini 2.0 Flash"
- Instructor docs and blog

### Papers
- KIEval (2025): evaluation metric for document KIE
- Structured CoT (EMNLP 2024 Findings)
- PLoP 2024: Prompt Patterns for Structured Data Extraction
- Benchmarking LLM-based IE tools for medical documents (2026)

### Tools and libraries
- Instructor: python.useinstructor.com
- Zerox: github.com/getomni-ai/zerox
- Docling: github.com/docling-project/docling
- PyMuPDF4LLM: pymupdf.readthedocs.io
- OpenRouter: openrouter.ai
- LLMWhisperer: unstract.com
