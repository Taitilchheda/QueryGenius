from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from fastapi.testclient import TestClient

import src.api as api_module
from src.api import app
from src.ingest import ingest_documents
from src.rag import RAGEngine
from src.utils import BASE_DIR


def _prepare_index() -> RAGEngine:
    os.environ["QG_USE_HASH_EMBEDDINGS"] = "1"
    os.environ["QG_ENABLE_LLM"] = "0"
    temp = TemporaryDirectory()
    tmp_root = Path(temp.name)
    raw_test = tmp_root / "raw_test"
    processed = tmp_root / "processed"
    index = tmp_root / "index"
    raw_test.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    index.mkdir(parents=True, exist_ok=True)
    (raw_test / "mini_algebra.txt").write_text(
        "A variable is a symbol. A linear equation can be ax + b = c.",
        encoding="utf-8",
    )
    (raw_test / "mini_python.txt").write_text(
        "A Python list stores an ordered mutable collection of items.",
        encoding="utf-8",
    )
    chunks_path = processed / "chunks.jsonl"
    index_path = index / "faiss.index"
    meta_path = index / "metadata.json"
    ingest_documents(
        raw_dir=raw_test,
        rebuild=True,
        chunks_path=chunks_path,
        index_path=index_path,
        index_meta_path=meta_path,
    )
    engine = RAGEngine(
        index_path=index_path,
        metadata_path=meta_path,
        chunks_path=chunks_path,
        enable_llm=False,
    )
    engine._tempdir = temp  # keep temp dir alive for test duration
    return engine


def test_retrieve_returns_chunks() -> None:
    engine = _prepare_index()
    hits = engine.retrieve("What is a linear equation?", top_k=3)
    assert hits
    sources = [Path(h.chunk.source_path).name for h in hits]
    assert any(name in {"mini_algebra.txt", "mini_python.txt"} for name in sources)


def test_ask_endpoint_fallback() -> None:
    engine = _prepare_index()
    api_module._engine = engine
    api_module.get_engine = lambda refresh=False: engine
    client = TestClient(app)
    response = client.post("/ask", json={"question": "What does a Python list store?", "top_k": 3})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "answer" in payload
    assert isinstance(payload["citations"], list)
    assert payload["latency_ms"] >= 0


def test_upload_and_chat_history() -> None:
    engine = _prepare_index()
    api_module._engine = engine
    api_module.get_engine = lambda refresh=False: engine
    client = TestClient(app)

    file_name = f"tmp_{uuid4().hex}.txt"
    upload = client.post(
        "/upload",
        files={"files": (file_name, b"A variable can represent a number in algebra.", "text/plain")},
    )
    assert upload.status_code == 200, upload.text
    assert file_name in upload.json()["uploaded"]

    session_id = f"session-{uuid4().hex}"
    chat = client.post(
        "/chat",
        json={"question": "What is a variable?", "top_k": 3, "session_id": session_id},
    )
    assert chat.status_code == 200, chat.text
    body = chat.json()
    assert body["session_id"] == session_id
    assert "answer" in body

    history = client.get(f"/history?session_id={session_id}&limit=10")
    assert history.status_code == 200, history.text
    messages = history.json()["messages"]
    assert any(m["session_id"] == session_id for m in messages)
