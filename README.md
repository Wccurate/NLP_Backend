# Simple RAG + Agent Backend

Minimal FastAPI backend that demonstrates a single-session career assistant with LangChain + LangGraph routing, OpenAI-powered generation, and hybrid retrieval over Chroma + BM25.

## Features
- Intent recognition with OpenAI zero-shot or TF‑IDF/LogReg fallback.
- Tools for normal chat, mock interviews, resume critique (JSON), and job matching with HyDE-assisted hybrid RAG.
- SQLite history table and sliding window memory.
- Optional web search enrichment via DuckDuckGo.
- Streaming or JSON responses from `/generate`.

## Requirements
- Python 3.10+
- An OpenAI API key (`OPENAI_API_KEY`).

Install OS dependencies (Mac/Linux):
```bash
conda create -n nlp_backend python=3.11 -y && conda activate nlp_backend
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
1. Copy `.env.example` to `.env`.
2. Fill in `OPENAI_API_KEY` and tweak other settings as needed:
   - `MODEL_NAME` (default `gpt-4o-mini`)
   - `TEMPERATURE`
   - `HISTORY_WINDOW`
   - `CHROMA_DIR`
   - `PRIMARY_INTENT_MODE` (`openai` or `fallback`)

The `scripts/dev_up.sh` script sources `.env` automatically.

## Running the API
```bash
./scripts/dev_up.sh
```

This boots `uvicorn` with autoreload on `http://0.0.0.0:8000`.

Key endpoints:
- `GET /health` – service heartbeat.
- `GET /history?limit=20` – last conversation turns.
- `POST /generate` – main interaction entrypoint.

Example request:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"input":"Match me with remote NLP roles","web_search":false}'
```

### Streaming responses
Set `"return_stream": true` to receive newline-delimited text chunks.

### Document ingestion
Include `<document>...</document>` in the `input` to index transient text snippets. Set `persist_documents=true` to keep them in Chroma; otherwise they are removed after the request.

## Testing
```bash
pytest
```
The suite uses `TestClient` and monkeypatches the OpenAI provider for the `/generate` happy path test.

## Docker (optional)
```bash
docker compose up --build
```
The API becomes available on port `8000`, sharing the project folder and `./data` volume with the container.

## Project Structure
- `app/main.py` – FastAPI app wiring, request handling, streaming.
- `app/agents/` – LangGraph router and intent-specific tool logic.
- `app/rag/` – Chroma + BM25 hybrid retrieval and HyDE helper.
- `app/tools/` – Intent classifier and async web search utility.
- `app/utils/` – Text parsing and sliding window helpers.
- `tests/` – Health and `/generate` route tests.

## Notes
- The backend stores conversation history in `data/app.db`. Remove the file to reset state.
- Chroma embeddings live under `data/chroma/`. Delete the directory to clear the vector store.
- When `PRIMARY_INTENT_MODE=fallback`, OpenAI is only used for generation/embeddings (intent falls back to the logistic regression classifier).
