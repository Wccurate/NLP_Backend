# Data Storage Overview

The backend uses two persistence layers:

1. **SQLite (relational)** – stores the conversational history in `./data/app.db`.
2. **ChromaDB (vector)** – persists job description embeddings in `./data/chroma/`.

---

## SQLite (`./data/app.db`)

- **Engine**: SQLite via SQLAlchemy (`sqlite:///data/app.db`).
- **Schema**: Single table named `history`.

### Table: `history`

| Column      | Type       | Constraints        | Description                                        |
|-------------|------------|--------------------|----------------------------------------------------|
| `id`        | INTEGER    | Primary Key        | Auto-incrementing identifier.                      |
| `role`      | TEXT       | Not Null, Indexed  | Speaker role, either `"user"` or `"assistant"`.    |
| `content`   | TEXT       | Not Null           | Message text including appended `<document>` tags when present. |
| `intent`    | TEXT       | Nullable           | Detected intent for the turn (`normal_chat`, `mock_interview`, `evaluate_resume`, `recommend_job`). |
| `created_at`| TIMESTAMP  | Not Null, Indexed  | UTC timestamp of message creation (`datetime.utcnow`). |

- **Data Flow**:
  - Each `/generate` call writes two rows: the combined user input (question + `<document>` block when files are uploaded) and the assistant reply.
  - `/history` reads this table ordered by `created_at`.

- **Why SQLite?**
  - Lightweight, serverless, and fits the single-session demo requirement.
  - Easy to inspect (`sqlite3 data/app.db`) and reset (delete the file).

---

## Chroma Vector Store (`./data/chroma/`)

- **Engine**: ChromaDB persistent client with collection `jobs_demo`.
- **Documents**: Each job description chunk added through `/jobs` becomes a document with metadata.

### Stored Fields per Document

| Field    | Description |
|----------|-------------|
| `id` | Chroma-generated UUID (or provided) referencing the chunk. |
| `documents` | The text chunk derived from the original job description. |
| `embeddings` | Vector representation generated via OpenAI embeddings (`text-embedding-3-small`). |
| `metadatas` | Dict with keys such as `source`, `title`, `type`, `created_at`, and any user-supplied metadata. |

- **Retrieval**:
  - `/generate` uses the retriever only for `recommend_job` intents.
  - Hybrid search (`vector_store.search`) blends dense similarity (OpenAI embeddings) with optional BM25 fallback.
  - The integrated HyDE step generates a synthetic answer to improve initial query embeddings.
- **Seeding**: During app startup, the bundled `_DEFAULT_CORPUS` is inserted into the collection if it is empty, ensuring BM25 and dense search share the same corpus.

- **Why Chroma?**
  - Provides a simple, file-based vector store with persistent collections, perfect for quick demos.
  - Integrates smoothly with LangChain and allows future migration to other stores (e.g., Qdrant).

- **Adding Data**:
  - Use `POST /jobs` to index new job descriptions; uploaded resumes are not stored here.

- **Clearing Data**:
  - Delete the `./data/chroma/` directory to reset the vector store.

---

## Configuration Notes

- Paths can be configured via environment variables (`CHROMA_DIR`, etc.) in `.env`.
- Ensure the process has write access to `./data/` for both SQLite and Chroma.
