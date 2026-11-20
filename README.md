# Medical Notes Helper

A fast, privacy-first CLI tool that helps clinic staff draft consistent administrative sections of medical notes by generating real-time next-word suggestions from an **n-gram language model built from scratch**.

Built for the **askThili — Build a Next-Word Predictor From Scratch** healthcare project.

---

## Features

- **Anonymized Administrative Medical Corpus (`app/corpus.py`, `data/medical_corpus.txt`)**
  - Curated corpus of 145+ realistic, 100% anonymized administrative medical notes (intake, scheduling, referrals, prior authorizations, insurance verification, billing, discharge coordination, and telehealth).
  - Built-in PHI anonymization pipeline that automatically redacts emails, phone numbers, dates, SSNs, MRNs, and titled names into safe placeholders (`<email>`, `<phone>`, `<date>`, `<id>`, `<ssn>`, `<name>`).
- **From-Scratch N-Gram Language Model with Multiple Smoothing Algorithms (`app/core.py`)**
  - Configurable n-gram order (default: trigram with automatic bigram/unigram backoff).
  - **Interpolated Kneser-Ney Smoothing** (default, lowest held-out perplexity).
  - **Laplace (Add-1) Smoothing** and **Add-k (Lidstone) Smoothing**.
  - Built-in held-out **perplexity benchmark** (`evaluate`) to compare smoothing algorithms quantitatively.
- **Next-Word Suggestion & Phrase Completion (`app/core.py`)**
  - Ranked next-word predictions with normalized confidence percentages and matched n-gram order indicators.
  - Configurable suggestion count (`--top-k`) and multi-word administrative phrase auto-completion (`generate`).
- **Interactive Drafting CLI & Feedback Loop (`app/cli.py`, `app/storage.py`)**
  - Interactive `draft` mode: type text, press a number (`1..N`) to accept a suggestion, `:reject <#>` to downvote unhelpful predictions, `:complete` to finish a sentence, `:k <N>` to change suggestion count live, `:smoothing <mode>` to swap smoothing live, and `:save [title]` to store completed drafts.
  - **Adaptive Feedback**: Accepted and rejected suggestions are persisted locally so future predictions adapt to your clinic's phrasing.
  - **Custom Clinic Phrases**: Add clinic-specific boilerplate phrases via `add-phrase` to immediately expand the model's vocabulary.
  - Runs **100% offline on your laptop** with zero API keys required (optional Ollama/Groq/Gemini config supported via `.env`).

---

## Tech Stack

- **Language:** Python 3.12+
- **Core NLP:** Pure Python standard library (`collections`, `math`, `re`) — zero ML framework black boxes
- **CLI UI:** `argparse` + `rich` (with `--plain` mode for scripts and CI)
- **Config & Persistence:** `python-dotenv` + local JSON storage
- **Testing & CI:** `pytest` + GitHub Actions (`.github/workflows/ci.yml`)

---

## Project Structure

```text
medical-notes-helper/
├── .env.example
├── .github/workflows/ci.yml
├── .gitignore
├── README.md
├── pytest.ini
├── requirements.txt
├── data/
│   └── medical_corpus.txt   # Anonymized administrative medical notes corpus
├── app/
│   ├── __init__.py
│   ├── corpus.py            # PHI anonymization, cleaning, sentence & word tokenization
│   ├── core.py              # N-gram model (Laplace, Add-k, Kneser-Ney) & text generation
│   ├── storage.py           # Local JSON persistence for feedback, custom phrases & drafts
│   ├── llm.py               # Optional LLM note-polishing adapter (disabled by default)
│   └── cli.py               # Interactive CLI entrypoint
└── tests/
    ├── test_core.py         # Unit tests for corpus, smoothing, prediction & feedback
    └── test_smoke.py        # End-to-end CLI smoke tests
```

---

## Quickstart

### 1. Set up a virtual environment and install dependencies

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. Copy the environment settings (optional)

```bash
cp .env.example .env
```

*(No API keys are needed—the n-gram language model runs completely locally. Never commit `.env`.)*

### 3. Run the CLI

```bash
# Welcome banner & corpus summary
python -m app.cli hello

# Get next-word suggestions (default: trigram + Kneser-Ney smoothing, top 5)
python -m app.cli suggest "the patient arrived"

# Customize number of suggestions and smoothing algorithm
python -m app.cli suggest "prior authorization" --top-k 8 --smoothing laplace

# Auto-complete an administrative note phrase
python -m app.cli generate "please schedule a follow-up" --words 10

# Launch interactive drafting studio
python -m app.cli draft

# Teach the model your clinic's preferred phrasing
python -m app.cli feedback --context "follow up" --word "appointment" --accept
python -m app.cli add-phrase "Insurance co-pay was collected at the front desk."

# View corpus, n-gram, and feedback statistics
python -m app.cli stats

# Benchmark Laplace vs Add-k vs Kneser-Ney on held-out perplexity
python -m app.cli evaluate
```

### 4. Run the test suite

```bash
pytest -v
```

---

## Milestones Completed

- [x] **Milestone 1:** Curated an anonymized corpus of administrative medical text with PHI redaction, text cleaning, and padded sentence tokenization.
- [x] **Milestone 2:** Implemented a from-scratch n-gram language model with Laplace, Add-k, and Interpolated Kneser-Ney smoothing.
- [x] **Milestone 3:** Built ranked next-word suggestion and multi-word phrase completion using conditional n-gram probabilities.
- [x] **Milestone 4:** Created a polished CLI with both one-shot commands and an interactive note-drafting loop.
- [x] **Stretch Goals:** Added Kneser-Ney smoothing + perplexity evaluation, persistent user feedback boosting/penalizing, custom clinic phrases, and configurable `--top-k` suggestion counts.
