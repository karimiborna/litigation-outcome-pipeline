# Deep Dive: Label Extraction & PDF Pipeline

**Date**: 2026-04-16
**Files analyzed**:
- [features/labels.py](../features/labels.py)
- [notebooks/colab_gpu_extraction.ipynb](../notebooks/colab_gpu_extraction.ipynb)

---

## Overview

This code forms the **ground truth layer** of the ML pipeline. Before you can train any model to predict case outcomes, you need two things:

1. **The documents** — PDFs downloaded from the SF Superior Court website
2. **The labels** — structured answers to "who won, and how much?"

This phase does both. The notebook handles downloading and text extraction on a GPU in the cloud (Colab). `features/labels.py` uses an LLM to read the extracted text and produce structured labels that become the `y` (target) in your ML training.

---

## Code Walkthrough

### `features/labels.py`

#### The keyword filter (`LABEL_DOC_KEYWORDS`)

```python
LABEL_DOC_KEYWORDS = {
    "JUDGMENT", "ORDER", "DISMISSAL",
    "Notice_of_Entry_of_Judgment", "STIPULATION", "COURT_JUDGMENT",
}
```

**What**: A set of strings used to filter which documents contain outcome information.

**Why**: Not every PDF in a case folder is relevant for labels. Filing forms, evidence exhibits, and procedural notices don't tell you who won — only judgment-type documents do. This filter prevents wasting LLM API tokens on irrelevant documents and reduces noise in the extracted labels.

**When to use this pattern**: Whitelist filtering is the right call when:
- You're dealing with heterogeneous document collections
- Some documents are expensive to process (LLM calls cost money)
- The signal-to-noise ratio matters for downstream quality

**Alternative**: You could send *all* documents to the LLM and let it figure out relevance — but that's 3-5x more expensive and gives the LLM more junk to confuse it.

---

#### The `CaseLabels` Pydantic model

```python
class CaseLabels(BaseModel):
    outcome: str | None = Field(None, description="plaintiff_win, ...")
    amount_awarded_principal: float | None = None
    defendant_appeared: bool | None = None
    ...
```

**What**: A typed container for the structured data extracted from each case.

**Why Pydantic**: Pydantic enforces types at runtime — if the LLM returns `"yes"` for a boolean field, Pydantic catches it. This is critical when your data source is an LLM, because LLMs don't always follow instructions perfectly.

**The `| None` on everything**: Every field is optional (`None` means "unclear or missing"). This is deliberate — the code prioritizes honesty. A `null` in the dataset is much better than a hallucinated value, because you can filter nulls out at training time but you can't un-corrupt a bad label.

**The `total_awarded` property**:
```python
@property
def total_awarded(self) -> float | None:
    parts = [self.amount_awarded_principal, self.amount_awarded_costs, ...]
    if all(p is None for p in parts):
        return None
    return sum(p or 0.0 for p in parts)
```

**What**: Computes total money awarded by summing the three components.

**Why a `@property` not a stored field**: It's derived data — there's no value in storing it separately. If any of the components change, the total stays correct automatically. It also keeps the LLM prompt simpler (3 fields instead of 4).

**The `all(p is None ...)`** check: Returns `None` (not `0.0`) when all components are unknown. This preserves the "uncertain" signal — a case with no amount vs. a case with $0 awarded are different things.

---

#### The prompt design

```python
LABEL_EXTRACTION_SYSTEM = (
    "You are a legal analyst AI... Respond with ONLY valid JSON — "
    "no explanation, no markdown, no extra text."
)
```

**What**: The system prompt that defines the LLM's role and output format.

**Why be so strict about "ONLY valid JSON"**: LLMs tend to wrap JSON in markdown code blocks (` ```json `) or add explanation text like "Here is the extracted data:". That breaks `json.loads()`. Being explicit in the prompt dramatically reduces parsing failures.

**Why `temperature=0.0`**:
```python
temperature=0.0
```
Temperature controls randomness. At 0.0, the LLM always picks the highest-probability next token — making outputs deterministic and consistent. For structured extraction (not creative writing), you want zero variance. The same document should always produce the same label.

**Why `response_format={"type": "json_object"}`**:
This is an OpenAI-specific feature that forces the model to output valid JSON — it's a second layer of defense on top of the prompt instruction.

---

#### Content-addressable caching

```python
def _cache_key(self, case_number: str, outcome_text: str) -> str:
    content = f"{case_number}:{outcome_text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

**What**: Generates a unique ID for a (case, document text) pair, used as the cache filename.

**Why hash the content, not just the case number**: If the extracted text changes (e.g., a better PDF extraction produces more text), the hash changes and the cache is invalidated automatically. Using just the case number as the key would serve stale cached labels even after you re-ran extraction.

**Why SHA-256**: It's a cryptographic hash — essentially zero collision probability for inputs of this size. MD5 would work too, but SHA-256 is the default good choice.

**Why `[:16]`**: 16 hex characters = 64 bits of entropy. That's more than enough for a cache key with thousands of files. Keeping filenames short makes them readable in a directory listing.

**The pattern (content-addressable storage)**: This is the same idea behind Git's object store, Docker layers, and npm's lockfile cache. You derive a key from the *content*, so the cache automatically handles invalidation.

---

#### Smart truncation

```python
@staticmethod
def _smart_truncate(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 3   # ~2,667 chars from the start
    tail = max_chars - head  # ~5,333 chars from the end
    return text[:head] + "\n\n[... middle truncated ...]\n\n" + text[-tail:]
```

**What**: Clips long documents to fit within LLM context limits while keeping the most useful parts.

**Why keep more from the tail**: Court judgments put the *ruling* at the end ("THEREFORE, judgment is entered for plaintiff in the amount of..."). The beginning has boilerplate headers and party info. Keeping 2/3 of the budget at the end captures the actual outcome more reliably.

**Why not just truncate from the end**: That would throw away the ruling entirely for long documents.

**Alternative**: Chunk the document and run multiple LLM calls, then merge. More accurate but 3-5x more expensive. Smart truncation is a reasonable cost/quality tradeoff at this scale.

---

### `notebooks/colab_gpu_extraction.ipynb`

#### Two-pass extraction strategy

The notebook implements a two-pass PDF extraction:

**Pass 1 — PyMuPDF (fast, free)**:
```python
import fitz  # PyMuPDF
def extract_with_pymupdf(pdf_path):
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    return "\n\n".join(pages).strip()
```
PyMuPDF reads the text layer embedded in the PDF. This works instantly for digitally-created PDFs.

**Pass 2 — Vision LLM (slow, expensive)**:
Only used for scanned documents where PyMuPDF returns less than `MIN_TEXT_LENGTH = 100` characters. These are images disguised as PDFs — the text layer is empty, so you need a vision model to "read" the pixels.

**Why this matters**: The design avoids paying for GPU inference on documents that don't need it. Court PDFs are a mix — some are digital forms, some are scanned paper documents. The 100-character threshold catches the scanned ones.

**The NVIDIA API path**:
```python
NVIDIA_MODEL = "meta/llama-3.2-90b-vision-instruct"
NVIDIA_DAILY_CAP = 490  # stay under the 500/day limit
```
NVIDIA offers free API access (~500 calls/day per key). A 90B vision model is significantly more capable than smaller models for reading messy scanned court documents.

#### Team work splitting

```python
MY_WORKER = 0       # 0-indexed — each team member picks a unique number
NUM_WORKERS = 3     # total team size
cases_to_process = all_cases[MY_WORKER::NUM_WORKERS]
```

**What**: Divides cases across team members using array slicing with a step.

**Why `[MY_WORKER::NUM_WORKERS]`**: This is interleaved sharding (not block sharding). Worker 0 gets cases 0, 3, 6... Worker 1 gets 1, 4, 7... Worker 2 gets 2, 5, 8...

Interleaved is better than blocks (`[:1/3]`, `[1/3:2/3]`, `[2/3:]`) because case difficulty tends to correlate with age or case number. Interleaving distributes hard/easy cases evenly across workers.

**Alternative**: A proper task queue (Celery, RQ, or even a shared Google Sheet of "claimed" cases). That's more robust but requires infrastructure. For a small team, modular arithmetic is enough.

---

## Concepts Explained

### Structured Extraction with LLMs

**The challenge**: LLMs output free text. ML models need numbers and categories. Structured extraction bridges this gap.

**The pattern**:
1. Write a prompt that asks for JSON with specific fields
2. Parse the response with `json.loads()`
3. Validate with Pydantic
4. Handle failures gracefully (log, return `None`, don't crash)

**Why this is separate from prediction**: LLMs are good at reading and parsing natural language. They're not calibrated for probabilistic predictions — they don't know their own uncertainty. So LLM = reading comprehension, ML model = prediction. This division is the architectural centerpiece of the whole project.

### Content-Addressable Caching

Same concept as Git objects, Docker layers, and Bazel's build cache. The key insight: if you can derive a unique ID from the *input content*, you never need to invalidate the cache manually — the hash does it for you.

### PDF Text Extraction Approaches

| Method | Speed | Cost | Works on |
|---|---|---|---|
| PyMuPDF (`fitz`) | Instant | Free | Digital PDFs |
| OCR (Tesseract) | Slow | Free | Scanned images |
| Vision LLM (NVIDIA) | Moderate | API cost | Scanned + complex layouts |

The notebook uses #1 first, falls back to #3. #2 was likely considered but vision LLMs are more accurate on messy court documents.

---

## Learning Resources

**Pydantic**:
- [Pydantic v2 docs](https://docs.pydantic.dev/latest/) — model validation, field types, custom validators
- [Pydantic tutorial: validation](https://docs.pydantic.dev/latest/concepts/models/) — how `model_validate()` works

**LLM structured output**:
- [OpenAI JSON mode docs](https://platform.openai.com/docs/guides/structured-outputs) — `response_format`, guaranteed JSON
- [Prompt engineering for extraction](https://www.promptingguide.ai/techniques/rag) — how to write reliable extraction prompts

**Hashing and content-addressable storage**:
- [Python `hashlib` docs](https://docs.python.org/3/library/hashlib.html)
- [Git internals: objects](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects) — the original content-addressable design

**PyMuPDF**:
- [PyMuPDF docs](https://pymupdf.readthedocs.io/en/latest/) — `fitz.open()`, `page.get_text()`

**Vision LLMs**:
- [LLaMA 3.2 Vision model card](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct) — what the NVIDIA model is

---

## Related Code in This Repo

- [features/extraction.py](../features/extraction.py) — same LLM pattern, but for *features* (not labels)
- [features/schema.py](../features/schema.py) — `LLMFeatures` Pydantic model (parallel to `CaseLabels`)
- [features/config.py](../features/config.py) — `FeaturesConfig` — where the API key and model name come from
- [scraper/court_api.py](../scraper/court_api.py) — upstream step that downloads the PDFs being processed here

---

## What to Study Next

1. **If you want to understand the feature extraction side**: Read [features/extraction.py](../features/extraction.py) — it's the same pattern but for richer case features (not just outcome labels)
2. **If you want to understand how labels become training data**: Look at [scripts/train_binary_classifier.py](../scripts/train_binary_classifier.py) — it loads labels and features together
3. **If you want to go deeper on LLM reliability**: Search for "LLM extraction reliability" and "structured outputs" — this is an active research area with papers on hallucination rates in extraction tasks
