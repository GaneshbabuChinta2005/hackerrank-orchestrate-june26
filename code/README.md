# HackerRank Orchestrate — Code Solution

Multi-modal damage claim verification system using Groq's Llama 4 Scout Vision LLM.

## What This Does

For each row in `dataset/claims.csv`, this system:
1. Loads the claim conversation, submitted images, user history, and evidence requirements
2. Sends all images + context to **Llama 4 Scout** via Groq's vision API
3. Receives a structured JSON verdict with all required output fields
4. Applies a post-VLM risk flag overlay from user history
5. Caches results to avoid repeated API calls
6. Writes the final `output.csv` to the repo root

## Files

| File | Purpose |
|---|---|
| `main.py` | Pipeline entry point — processes `claims.csv` → `output.csv` |
| `analyzer.py` | Gemini Vision API calls with retry + rate limiting |
| `prompts.py` | Prompt builder with prompt injection defenses |
| `config.py` | Allowed value lists, model name, rate limit settings |
| `risk.py` | Post-VLM user history risk flag overlay |
| `utils.py` | CSV I/O, image loading, output validation |
| `requirements.txt` | Python dependencies |
| `evaluation/main.py` | Evaluation pipeline on sample_claims.csv |
| `evaluation/evaluation_report.md` | Auto-generated evaluation report |

## Setup

### 1. Install dependencies

```bash
cd code
pip install -r requirements.txt
```

### 2. Set your Groq API Key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_actual_key
```

Get a free key at: https://console.groq.com/keys

### 3. Run the solution

```bash
# From the repo root or code/ directory:
python code/main.py
```

This reads `dataset/claims.csv`, processes all 45 claims, and writes `output.csv` to the repo root.

**Options:**
```bash
python code/main.py --input dataset/claims.csv --output output.csv
python code/main.py --no-cache   # Force re-process all rows (ignores cache)
```

### 4. Run evaluation (optional)

```bash
python code/evaluation/main.py
```

This runs the pipeline on `dataset/sample_claims.csv` (which has ground-truth labels), computes accuracy metrics, and writes `code/evaluation/evaluation_report.md`.

## Design Decisions

### Model: Llama 4 Scout via Groq
- Ultra-fast inference via Groq's LPU hardware
- Vision-capable: handles images natively as base64 data URIs
- Very low cost (free tier available, ~$0.01 for the full 45-claim test set)

### Rate Limiting
- 1.2-second delay between API calls → stays under 50 RPM
- Exponential backoff on errors (5s → 10s → 20s)

### Caching
- Results cached to `cache/results_cache.json`
- Cache key = user_id + image_paths + claim_object
- Re-runs are instant after first run

### Prompt Injection Defense
- Explicit instruction to ignore text in images saying "approve this claim", "skip review", etc.
- Explicit instruction to ignore such text in the claim conversation
- Several test cases contain these injections — they are flagged with `text_instruction_present`

### Risk Flag Overlay
- VLM identifies image-based risk flags (blurry, wrong object, claim mismatch, etc.)
- After VLM inference, user history flags are merged in (user_history_risk, manual_review_required)

## Output Schema

The `output.csv` contains these columns in order:

`user_id`, `image_paths`, `user_claim`, `claim_object`, `evidence_standard_met`, `evidence_standard_met_reason`, `risk_flags`, `issue_type`, `object_part`, `claim_status`, `claim_status_justification`, `supporting_image_ids`, `valid_image`, `severity`
