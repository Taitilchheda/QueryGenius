# QueryGenius v2

QueryGenius is a local-first RAG assistant for PDF/TXT/MD Q&A with citations, diagram rendering, and LaTeX-style formula rendering.

## Screenshots

![Formula Rendering](docs/screenshots/querygenius-formula.png)
![Diagram + Formula Answer](docs/screenshots/querygenius-gan-diagram.png)

## Core Features

- Ingest local docs: `.txt`, `.md`, `.pdf`
- Chunk + embed with `sentence-transformers/all-MiniLM-L6-v2`
- FAISS local vector index with disk persistence
- FastAPI backend + modern web UI
- Source-grounded answers with citations (`source + chunk_id + score`)
- Diagram responses (Mermaid + fallback renderer + zoom on click)
- Formula responses rendered with KaTeX
- Chat sessions with archive/delete
- Local auth (register/login/logout)
- Evaluation script (Recall@1/3/5 + latency)

## Project Structure

```text
querygenius-v2/
  README.md
  requirements.txt
  .env.example
  data/
    raw/
    processed/
    index/
    eval/
  src/
    ingest.py
    rag.py
    api.py
    eval.py
    utils.py
    static/
      index.html
      styles.css
      app.js
  tests/
    test_rag.py
```

## Quick Start (Windows)

```powershell
cd querygenius-v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

## Add Documents and Build Index

1. Put files into `data/raw/`
2. Build/rebuild index:

```powershell
python -m src.ingest --rebuild
```

Generated artifacts:
- `data/processed/chunks.jsonl`
- `data/index/faiss.index`
- `data/index/metadata.json`

## Run App

```powershell
uvicorn src.api:app --reload
```

Open:
- `http://127.0.0.1:8000/`

## API (Important Endpoints)

- `GET /health`
- `GET /documents`
- `POST /upload`
- `POST /ingest`
- `POST /ask`
- `POST /chat`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `GET /chats`
- `POST /chats`
- `PATCH /chats/{session_id}`
- `DELETE /chats/{session_id}`

## Minimal cURL Examples

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is self-attention?","top_k":5}'
```

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"give the diagram of GAN with formula and logic","top_k":5}'
```

## Evaluation

```powershell
python -m src.eval
```

Uses:
- `data/eval/eval_questions.json`

Outputs:
- Recall@1 / Recall@3 / Recall@5
- avg retrieval/generation/total latency
- `data/eval/report.json`

## Essential Config (`.env`)

- `QG_EMBEDDING_MODEL`
- `QG_EMBEDDING_DEVICE` (`cuda` or `cpu`)
- `QG_ENABLE_LLM`
- `QG_LLM_MODEL`
- `QG_MAX_NEW_TOKENS`
- `QG_MAX_NEW_TOKENS_DIAGRAM`
- `QG_STRICT_GROUNDED`
- `QG_ENFORCE_SOURCE_FOCUS`

## GPU Check

```bash
curl http://127.0.0.1:8000/health
```

Look for:
- `"cuda_available": true`

## Troubleshooting

- Wrong references: run `python -m src.ingest --rebuild`
- Formula not rendered: restart API + hard refresh (`Ctrl+F5`)
- Diagram not rendered: Mermaid CDN issue; fallback should still show
- Slow responses: verify CUDA, reduce `top_k`, reduce `QG_MAX_NEW_TOKENS`

## Test

```powershell
pytest -q
```
