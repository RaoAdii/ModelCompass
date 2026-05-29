# ModelCompass

Multi-model document summarization platform with a Flask API and a Vite + React UI. It routes documents to BART, Pegasus, and T5, scores summaries with ROUGE, and tracks analytics with a SQLite-backed cache.

## Highlights

- Upload PDF or TXT files and auto-detect document type when not provided.
- Generate three summaries and return a recommended summary from the routed model.
- ROUGE-1/2/L evaluation plus custom metrics (word count and compression ratio).
- Cache summaries and track analytics (latency, model usage, doc types) via API v2.
- React UI with drag-and-drop upload, side-by-side comparisons, and analytics dashboard.

## Project Structure

```text
ModelCompass/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── main.py
│   ├── db/
│   │   ├── __init__.py
│   │   └── schema.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── summarizer.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── api_v2.py
│   └── utils/
│       ├── __init__.py
│       ├── cache.py
│       └── text_processor.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── main.jsx
│       ├── components/
│       │   ├── SummaryComparison.css
│       │   └── SummaryComparison.jsx
│       └── pages/
│           └── AnalyticsDashboard.jsx
├── tests/
│   ├── conftest.py
│   ├── test_analytics.py
│   ├── test_api.py
│   ├── test_cache.py
│   ├── test_evaluator.py
│   └── test_summarizer.py
├── .env.example
└── requirements.txt
```

## Backend Quickstart

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a local `.env`:

```bash
# Windows
copy .env.example .env
# macOS/Linux
cp .env.example .env
```

4. Run the Flask API:

```bash
python -m app.main
```

Health check:

```bash
curl http://localhost:5000/api/health
```

## Frontend Quickstart

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Optionally configure the API base URL:

```bash
# Windows
set VITE_API_BASE_URL=http://localhost:5000
# macOS/Linux
export VITE_API_BASE_URL=http://localhost:5000
```

Or set it in `frontend/.env.local`:

```bash
VITE_API_BASE_URL=http://localhost:5000
```

3. Start the UI:

```bash
npm run dev
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEBUG` | `false` | Flask debug mode |
| `HOST` | `0.0.0.0` | API bind address |
| `PORT` | `5000` | API port |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `MAX_FILE_SIZE_MB` | `10` | Max upload size |
| `MODEL_TIMEOUT_SECONDS` | `60` | Summarization timeout |
| `MAX_INPUT_TOKENS` | `1024` | Input truncation limit |
| `BART_MODEL_NAME` | `facebook/bart-large-cnn` | BART model |
| `PEGASUS_MODEL_NAME` | `google/pegasus-xsum` | Pegasus model |
| `T5_MODEL_NAME` | `t5-small` | T5 model |
| `SUMMARY_MAX_LENGTH` | `180` | Summary max length |
| `SUMMARY_MIN_LENGTH` | `60` | Summary min length |
| `USE_PARALLEL_ON_CPU` | `false` | Parallel mode on CPU |
| `USE_PARALLEL_ON_GPU` | `true` | Parallel mode on GPU |
| `DB_PATH` | `app/db/cache.db` | SQLite DB path |

## API Reference

### `GET /api/health`

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

### `POST /api/v2/summarize`

Same form fields as v1. Returns cache and performance metadata:

```json
{
  "summaries": { "summary_bart": "..." },
  "rouge_scores": { "pairwise_rouge": { "summary_bart_vs_summary_t5": { "rouge1": 0.31 } } },
  "metadata": {
    "detected_doc_type": "news",
    "routed_model": "bart",
    "was_cached": false,
    "inference_time_ms": 1234.56,
    "doc_length_chars": 2048,
    "doc_hash": "...",
    "recommended_summary_key": "summary_bart"
  }
}
```

### `GET /api/v2/analytics`

Returns aggregate analytics, cache hit rate, and recent summaries.

### `POST /api/v2/cache/clear`

Clears all cache entries or a specific hash:

```json
{ "doc_hash": "optional" }
```

## Testing

Run the full test suite:

```bash
pytest -q
```

Verbose output:

```bash
python -m pytest tests -v --tb=short
```

## Performance Notes

- Timeout is controlled by `MODEL_TIMEOUT_SECONDS` and returns HTTP `504` on expiry.
- Parallel mode can improve GPU throughput but may not help on CPU for large models.
- Benchmark with your target documents and set timeouts to at least 2x p95 latency.

## Roadmap

- Model fine-tuning and domain-specific routing.
- Authentication, usage quotas, and audit logs.
- Extended analytics dashboards and export.
