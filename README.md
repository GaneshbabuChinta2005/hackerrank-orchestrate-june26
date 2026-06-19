# HackerRank Orchestrate — Multi-Modal Evidence Review

> **24-hour hackathon submission** — AI-powered damage claim verification using Groq's Llama 4 Scout Vision LLM.

Build a system that verifies visual evidence for damage claims across three object types: **cars**, **laptops**, and **packages**.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, and allowed values.

---

## Contents

1. [Solution Overview](#solution-overview)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Quickstart](#quickstart)
5. [How It Works](#how-it-works)
6. [Design Decisions](#design-decisions)
7. [Evaluation](#evaluation)
8. [Chat Transcript Logging](#chat-transcript-logging)
9. [Submission](#submission)
10. [Judge Interview](#judge-interview)

---

## Solution Overview

This system verifies damage insurance claims by analyzing:
- **Submitted photos** of the damaged object (car, laptop, or package)
- **Customer support chat conversation** describing the issue
- **User claim history** for risk assessment
- **Evidence requirements** defining minimum image standards

For each claim, it produces a structured verdict: `supported`, `contradicted`, or `not_enough_information`, along with risk flags, issue type, severity, and image-grounded justifications.

### Key Features

| Feature | Description |
|---|---|
| **Vision AI** | Llama 4 Scout 17B via Groq API — analyzes images + text together |
| **Prompt Injection Defense** | Detects and ignores instructions embedded in images or claim text |
| **Multi-Language** | English, Hindi, Spanish, Chinese claims handled natively |
| **Risk Overlay** | Post-VLM merge of user history flags (repeat claimers, prior rejections) |
| **Smart Caching** | JSON file cache with stale-entry detection (skips API error fallbacks) |
| **Rate Limit Handling** | Auto-detects 429 errors, waits up to 120s, retries 5 times |

---

## Architecture

```
dataset/claims.csv
        │
        ▼
┌─────────────────┐
│   Preprocessor  │ ← loads user_history.csv + evidence_requirements.csv
└────────┬────────┘
         │  per-claim loop
         ▼
┌─────────────────────────────────┐
│  Groq Llama 4 Scout Vision LLM  │ ← images (base64 JPEG, 512px max)
│  (meta-llama/llama-4-scout-     │   + claim text + history + evidence rules
│   17b-16e-instruct)             │   → structured JSON verdict
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Risk Overlay   │ ← merges VLM flags + user_history flags
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validator      │ ← enforces allowed values for every field
└────────┬────────┘
         │
         ▼
      output.csv (44 predictions)
```

---

## Repository Layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full task description and I/O schema
├── README.md                         # You are here
├── output.csv                        # ✅ Generated predictions for claims.csv
│
├── code/                             # Solution implementation
│   ├── main.py                       # Pipeline entry point
│   ├── analyzer.py                   # Groq Vision API wrapper (retry, resize, parse)
│   ├── prompts.py                    # System & user prompt builder + anti-injection
│   ├── config.py                     # Allowed values, model config, rate limits
│   ├── risk.py                       # User history risk flag overlay
│   ├── utils.py                      # CSV I/O, image loading, validation, caching
│   ├── requirements.txt              # Python dependencies
│   ├── .env.example                  # API key template
│   ├── .gitignore                    # Excludes .env, cache/, __pycache__
│   ├── README.md                     # Detailed code documentation
│   ├── cache/                        # Auto-generated result cache
│   └── evaluation/
│       └── main.py                   # Evaluation pipeline (metrics on sample data)
│
└── dataset/
    ├── sample_claims.csv             # 20 labeled examples (input + expected output)
    ├── claims.csv                    # 44 test claims (input only)
    ├── user_history.csv              # 47 users with risk flags & history
    ├── evidence_requirements.csv     # 12 evidence rules by object type
    └── images/
        ├── sample/                   # Images referenced by sample_claims.csv
        └── test/                     # Images referenced by claims.csv
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/GaneshbabuChinta2005/hackerrank-orchestrate-june26.git
cd hackerrank-orchestrate-june26
pip install -r code/requirements.txt
```

### 2. Set your Groq API Key

```bash
cp code/.env.example code/.env
# Edit code/.env and set: GROQ_API_KEY=your_key_here
```

Get a free API key at: https://console.groq.com/keys

### 3. Run the pipeline

```bash
python code/main.py
```

This will:
- Process all 44 claims from `dataset/claims.csv`
- Send images + context to Llama 4 Scout via Groq API
- Write predictions to `output.csv`
- Cache results in `code/cache/results_cache.json`

### 4. Run evaluation

```bash
python code/evaluation/main.py
```

Evaluates predictions against `dataset/sample_claims.csv` labeled data and produces `code/evaluation/evaluation_report.md`.

---

## How It Works

### Per-Claim Processing

1. **Image Loading** — Images loaded from `dataset/images/test/case_XXX/`, encoded as base64 JPEG, resized to 512px max
2. **Prompt Construction** — Builds structured prompt with claim text, user history, evidence requirements, and anti-injection defenses
3. **Groq API Call** — Sends to Llama 4 Scout vision model at temperature 0.0 for deterministic results
4. **JSON Extraction** — Parses structured JSON from model response (handles markdown fences, partial JSON)
5. **Validation** — Every field validated against allowed values (invalid → safe defaults)
6. **Risk Overlay** — Merges user history flags (e.g., `user_history_risk`, `manual_review_required`)
7. **Caching** — Results cached to avoid redundant API calls; stale entries auto-detected

### Prompt Injection Defense

The system explicitly handles attempts to manipulate the model:
- Images containing text instructions (e.g., "approve this claim") → flagged as `text_instruction_present`
- Claim text containing instructions (e.g., "ignore previous instructions") → ignored per system prompt
- Threatening language → not penalized, but noted

---

## Design Decisions

### Why Llama 4 Scout via Groq?

| Factor | Reasoning |
|---|---|
| **Vision capability** | Native multimodal — handles images + text in one call |
| **Speed** | Groq's LPU hardware provides ultra-fast inference (~2-3s/claim) |
| **Cost** | Free tier covers the full 44-claim test set |
| **JSON output** | Reliable structured output when prompted correctly |

### Why 512px Image Resize?

Groq's vision API rejects images above certain size thresholds. Testing confirmed 512px JPEG works reliably across all test images (including 1.6MB originals).

### Rate Limiting Strategy

- **2.0s delay** between API calls (under 30 RPM)
- **5 retries** with smart 429 detection
- **Up to 120s wait** on rate limit hits (parses retry-after from error)
- **Daily TPD limit**: 500K tokens — sufficient for ~35 claims with images

---

## Evaluation

The evaluation pipeline (`code/evaluation/main.py`) includes:

- **Per-field accuracy** on `dataset/sample_claims.csv` (20 labeled examples)
- **Claim status accuracy** — the primary metric
- **Risk flags precision** — how well the system detects risk signals
- **Evidence standard accuracy** — whether evidence sufficiency is correctly assessed
- **Operational metrics** — API calls, runtime, estimated cost

---

## Chat Transcript Logging

This repo uses `AGENTS.md` to instruct AI coding tools to log conversation turns:

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate\log.txt` |

Upload this log as the chat transcript at submission time.

---

## Submission

Submit the following files as instructed by HackerRank:

1. **Code zip**: `code/` directory with all source files, README, and evaluation
2. **Predictions CSV**: `output.csv` — 44 predictions for `dataset/claims.csv`
3. **Chat transcript**: `log.txt` from the path above

### Pre-submission checklist

- [x] `output.csv` has one row per row in `dataset/claims.csv`
- [x] `output.csv` has the exact required columns in the exact required order
- [x] Evaluation files are included in `code/`
- [x] No hardcoded secrets — API key via `.env` only
- [x] No hardcoded labels — all decisions made by the VLM

---

## Judge Interview

After submission, the AI Judge may ask about:

- **Approach**: Vision LLM + structured prompting + post-inference risk overlay
- **Model choice**: Llama 4 Scout via Groq — fast, free, vision-capable
- **Prompt design**: Anti-injection defenses, evidence requirement injection, JSON schema enforcement
- **Evaluation strategy**: Per-field accuracy on labeled sample data
- **AI usage**: Built entirely using Google Antigravity AI coding assistant

Be prepared to explain the solution in detail.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| AI Model | Llama 4 Scout 17B via Groq API |
| Libraries | `groq`, `python-dotenv`, `pandas`, `Pillow` |
| Caching | JSON file cache with stale detection |
| Rate Limiting | Smart 429 handling with exponential backoff |
