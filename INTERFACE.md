# API Interface Documentation

Base URL: `http://localhost:8000`

All JSON responses follow the `application/json` media type unless specified. Multipart form endpoints accept `multipart/form-data`.

---

## GET /health

**Summary**: Service heartbeat.

- **Request Body**: _None_
- **Response** `200 OK`:
  ```json
  {
    "status": "ok"
  }
  ```

---

## GET /history

**Summary**: Retrieve recent conversation turns.

- **Query Parameters**
  | Name | Type | Required | Default | Description |
  |------|------|----------|---------|-------------|
  | `limit` | integer (1–100) | No | `20` | Number of most recent entries to return. |

- **Response** `200 OK`:
  ```json
  [
    {
      "role": "user",
      "content": "Question text\n<document>\nDocument text\n</document>",
      "intent": "evaluate_resume",
      "timestamp": "2024-05-05T08:12:34.567890"
    },
    {
      "role": "assistant",
      "content": "Assistant reply...",
      "intent": "evaluate_resume",
      "timestamp": "2024-05-05T08:12:35.123456"
    }
  ]
  ```

---

## POST /generate

**Summary**: Primary conversation endpoint; routes intent to chat tools, mock interview, resume review, or job recommendation with RAG.

- **Content Type**: `multipart/form-data`

- **Form Fields**
  | Name | Type | Required | Default | Description |
  |------|------|----------|---------|-------------|
  | `input` | string | Yes\* | `""` | User question or message. Required unless a file is uploaded. |
  | `file` | file (DOCX/PDF optional text/plain) | No | — | Optional resume or supporting document. Extracted text is appended as `<document>…</document>` for processing and stored with the user turn. |
| `web_search` | boolean (`"true"`/`"false"`) | No | `false` | Deprecated. The agent now decides autonomously whether to search using MCP-style planning; this flag is ignored. |
  | `return_stream` | boolean | No | `false` | When `true`, the endpoint returns a streaming plain-text response (newline-delimited); otherwise JSON. |
  | `persist_documents` | boolean | No | `false` | Reserved flag; uploaded documents are not persisted to the vector store (job descriptions must be added via `/jobs`). |

- **Success Response** `200 OK` (non-streaming):
  ```json
  {
    "intent": "recommend_job",
    "text": "Recommended roles ... See [0], [1].",
    "sources": [
      {"source": "jobs_demo#42", "score": 0.87, "text": "ML Engineer role ..."},
      {"source": "jobs_demo#17", "score": 0.81, "text": "Data Scientist ..."}
    ],
    "tool_calls": ["recommend_job"]
  }
  ```

  - `intent` ∈ {`normal_chat`, `mock_interview`, `evaluate_resume`, `recommend_job`}
  - `sources` is populated for RAG responses; each item includes `dense_score` (embedding similarity), `bm25_score` (keyword similarity), `hybrid_score` (weighted blend), plus optional raw diagnostic values (`bm25_raw_score`, `dense_distance`).
  - `tool_calls` lists the tool executed in the agent graph. For normal chat, metadata includes `web_search` details when the agent invoked search.

- **Streaming Response** (when `return_stream=true`): `text/plain` newline-delimited chunks of the assistant reply.

- **Error Responses**
  - `422 Unprocessable Entity` – missing both `input` and file, or malformed request.
  - `500 Internal Server Error` – upstream LLM/RAG failures (message sanitized).

---

## POST /jobs

**Summary**: Index a job description for subsequent RAG queries.

- **Content Type**: `application/json`

- **Request Body**
  ```json
  {
    "text": "Senior NLP Engineer responsible for deploying LLM systems...",
    "title": "Senior NLP Engineer",
    "metadata": {
      "location": "Remote",
      "department": "AI"
    }
  }
  ```
  - `text` (string, required): Full job description content. Automatically chunked and embedded.
  - `title` (string, optional): Label used in metadata and generated source IDs.
  - `metadata` (object, optional): Additional key/value pairs (strings) stored alongside each chunk.

- **Response** `201 Created`:
  ```json
  {
    "inserted": 3,
    "ids": [
      "22b7c1a2-e21c-4b4f-b74a-483b4d2f9d18",
      "7f561386-0c18-42df-95fb-2e635b88caf3",
      "1cb2af86-e8e0-4dbc-8ea1-a1d8a1e6d1f7"
    ]
  }
  ```
  - `inserted`: Number of chunks added.
  - `ids`: Chroma document IDs for reference.

- **Error Responses**
  - `422 Unprocessable Entity` – missing/empty `text`.
  - `500 Internal Server Error` – vector store insertion failure.

---

## Notes on File Handling

- DOCX files are parsed using `python-docx`.
- PDFs use PyMuPDF (`fitz`) by default; if unavailable, fall back to PyPDF, with optional OCR via OCR.Space (`OCR_SPACE_API_KEY`).
- The extracted document text is appended to the stored user message in `<document>…</document>` blocks but is not persisted to the job vector store.

---

## Authentication

- No authentication is configured; this demo operates as a single-session system.
