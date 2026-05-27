# ModelCompass - MVP (Weeks 1-2)

Multi-model document summarization platform with Flask + React.

## Folder Structure

```text
ModelCompass/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── summarizer.py
│   ├── routes/
│   │   ├── __init__.py
│   │   └── api.py
│   └── utils/
│       ├── __init__.py
│       └── text_processor.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       └── components/
│           ├── SummaryComparison.css
│           └── SummaryComparison.jsx
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_evaluator.py
│   └── test_summarizer.py
├── .env.example
└── requirements.txt
```

## Features in This MVP

- Upload PDF/TXT documents via `POST /api/summarize`
- Optional `document_type` input (`research_paper`, `announcement`, `news`)
- Auto-detect document type when not provided
- Lazy-loaded HuggingFace models:
  - `facebook/bart-large-cnn`
  - `google/pegasus-xsum`
  - `t5-small`
- CPU/GPU-aware execution mode:
  - CPU default: sequential (`BART -> Pegasus -> T5`)
  - GPU default: parallel (`ThreadPoolExecutor`)
  - configurable with `USE_PARALLEL_ON_CPU` and `USE_PARALLEL_ON_GPU`
- 60s timeout budget with `504` timeout response mapping
- Pairwise ROUGE evaluation (`ROUGE-1`, `ROUGE-2`, `ROUGE-L`)
- Custom metrics (word count, compression ratio)
- React UI with:
  - drag-drop upload
  - loading spinner
  - side-by-side summaries
  - ROUGE bar chart (Recharts)
  - copy-to-clipboard buttons

## Backend Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from sample:

```bash
copy .env.example .env
```

4. Run Flask server:

```bash
python -m app.main
```

Health check:

```bash
curl http://localhost:5000/api/health
```

### Concurrency Mode

Set execution mode in `.env`:

```bash
USE_PARALLEL_ON_CPU=false
USE_PARALLEL_ON_GPU=true
```

- Use sequential mode on CPU for better stability/perf with large models.
- Use parallel mode on GPU for better throughput.

## API Contracts

### `GET /api/health`

Response:

```json
{
  "status": "ok",
  "timestamp_utc": "2026-05-21T18:00:00.000000+00:00"
}
```

### `POST /api/summarize`

Form fields:

- `file` (required): PDF or TXT
- `document_type` (optional): `research_paper` | `announcement` | `news`

Response shape:

```json
{
  "document_type": "news",
  "routed_model": "bart",
  "summaries": {
    "summary_bart": "...",
    "summary_pegasus": "...",
    "summary_t5": "..."
  },
  "recommended_summary": "...",
  "evaluation": {
    "pairwise_rouge": {
      "summary_bart_vs_summary_pegasus": {
        "rouge1": 0.31,
        "rouge2": 0.14,
        "rougeL": 0.28
      }
    },
    "custom_metrics": {
      "summary_bart": {
        "word_count": 68.0,
        "compression_ratio": 0.2
      }
    }
  }
}
```

## Frontend Setup

1. Install frontend dependencies:

```bash
cd frontend
npm install
```

2. Configure API base URL (optional):

```bash
set VITE_API_BASE_URL=http://localhost:5000
```

Or set in `frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://localhost:5000
```

3. Run frontend:

```bash
npm run dev
```

## Testing

Run all tests:

```bash
pytest -q
```

Included test coverage:

- `/api/health` success
- `/api/summarize` TXT and PDF upload flow with mocks
- invalid extension handling
- timeout behavior (`504`)
- evaluator sanity checks (`ROUGE > 0.1`)
- summarizer parallel/timeout behavior

Run verbose test mode:

```bash
python -m pytest tests/ -v --tb=short
```

## Timeout and Performance Notes

- API timeout is controlled by `MODEL_TIMEOUT_SECONDS` (default `60`).
- If timeout is reached, API returns HTTP `504` with a user-friendly error.
- CPU timing and GPU timing should be measured in your target deployment environment because throughput varies by hardware.

Suggested benchmark process:

1. Run once with `USE_PARALLEL_ON_CPU=false` on CPU-only machine.
2. Run once with GPU available and `USE_PARALLEL_ON_GPU=true`.
3. Record end-to-end latency for the same test document.
4. Keep timeout margin at least 2x observed p95 latency.

## Notes

- No fine-tuning logic included (Phase 3)
- No authentication included (Phase 4)
- No database included (future phase)
- Secrets are expected in `.env` only
