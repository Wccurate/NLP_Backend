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
- `POST /generate` – main interaction entrypoint (conversation + tools).
- `POST /jobs` – index a job description for hybrid retrieval.

### Terminal curl examples
1. Resume evaluation (text + PDF):
   ```bash
   curl -X POST http://localhost:8000/generate \
     -F "input=请帮我评估一下附件中的简历" \
     -F "file=@examples/WANGSHIBO_CV.pdf" \
     -F "persist_documents=false" \
     -F "return_stream=false"
   ```
2. Job recommendations (text only):
   ```bash
   curl -X POST http://localhost:8000/generate \
     -F "input=推荐一些远程的机器学习工程师岗位" \
     -F "web_search=false"
   ```
3. Mock interview practice:
   ```bash
   curl -X POST http://localhost:8000/generate \
     -F "input=我们来进行一次数据科学家的模拟面试" \
     -F "web_search=false"
   ```
4. Normal chat with optional web search:
   ```bash
   curl -X POST http://localhost:8000/generate \
     -F "input=谢谢，今天的建议很有帮助" \
     -F "web_search=true"
   ```
5. Add a job description to the vector store:
   ```bash
   curl -X POST http://localhost:8000/jobs \
     -H "Content-Type: application/json" \
     -d '{"text":"Senior NLP Engineer responsible for production LLM pipelines.","title":"Senior NLP Engineer","metadata":{"location":"Remote"}}'
   ```

### Streaming responses
Set `"return_stream": true` to receive newline-delimited text chunks.

### Document ingestion
- Upload Word/PDF files via the `file` field; text is extracted (with simple parsing) and wrapped in `<document>...</document>` before processing or storage.
- Users do not need to add `<document>` tags manually—the backend handles the wrapping automatically.
- Uploaded files are *not* added to the job vector store; only descriptions ingested through `POST /jobs` are indexed for RAG.
- PDF extraction defaults to a PyMuPDF-based parser; set `OCR_SPACE_API_KEY` (and optional `OCR_SPACE_ENDPOINT` / `OCR_SPACE_LANGUAGE`) to enable the OCR fallback strategy when needed.

## Testing
1. Install dependencies (once):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the entire test suite from the project root:
   ```bash
   PYTHONPATH=. pytest
   ```

Test layout:
- `tests/test_generate_api.py` – exercises `/generate` across resume evaluation, job matching, mock interview, and normal chat flows (uses `examples/WANGSHIBO_CV.pdf`).
- `tests/test_history_api.py` – verifies `/history` captures the combined text + `<document>` conversation turns.
- `tests/test_health.py` – quick smoke check for `/health`.

The fixtures in `tests/conftest.py` replace OpenAI and Chroma calls with deterministic stubs, so no real API key is required while running tests.

Run an individual test module:
```bash
PYTHONPATH=. pytest tests/test_history_api.py
```

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
