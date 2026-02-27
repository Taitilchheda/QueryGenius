from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from .ingest import ingest_documents
from .rag import RAGEngine
from .utils import PROCESSED_DIR, RAW_DIR, SUPPORTED_EXTENSIONS, ensure_data_dirs, read_jsonl

if load_dotenv is not None:
    load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="QueryGenius", version="0.2.0")
BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "src" / "static"
DB_PATH = PROCESSED_DIR / "app.db"
TOKEN_TTL_DAYS = 14

_engine: RAGEngine | None = None


class IngestRequest(BaseModel):
    rebuild: bool = Field(default=True)
    chunk_size_chars: int = Field(default=1500, ge=200)
    overlap_chars: int = Field(default=200, ge=0)


class IngestResponse(BaseModel):
    status: str
    num_chunks: int
    chunks_path: str
    index_path: str
    metadata_path: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_profile: str = Field(default="balanced")


class ChatRequest(AskRequest):
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    retrieval_profile: str
    latency_ms: float
    latency_breakdown_ms: dict[str, float]


class ChatResponse(AskResponse):
    session_id: str
    message_id: str
    question: str
    created_at: str


class UploadResponse(BaseModel):
    status: str
    uploaded: list[str]
    skipped: list[str]


class AuthRequest(BaseModel):
    email: str = Field(..., min_length=4)
    password: str = Field(..., min_length=6)


class AuthUser(BaseModel):
    id: int
    email: str


class AuthResponse(BaseModel):
    token: str
    user: AuthUser


class ChatCreateRequest(BaseModel):
    title: str | None = None


class ChatUpdateRequest(BaseModel):
    title: str | None = None
    archived: bool | None = None


class ChatSummary(BaseModel):
    session_id: str
    title: str
    turns: int
    archived: bool
    created_at: str
    last_updated: str


@contextmanager
def db_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    ensure_data_dirs()
    with db_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS chats (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_updated TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                citations_json TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                top_k INTEGER NOT NULL,
                retrieval_profile TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chats(session_id)
            );
            """
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    clean = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip().replace(" ", "_")
    if not clean:
        raise ValueError("Invalid filename.")
    return clean


def _password_hash(password: str, salt: str | None = None) -> str:
    use_salt = salt or secrets.token_hex(16)
    digest = hashlib.sha256((use_salt + password).encode("utf-8")).hexdigest()
    return f"{use_salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, old_digest = stored.split("$", 1)
    except ValueError:
        return False
    new_digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return secrets.compare_digest(new_digest, old_digest)


def _create_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=TOKEN_TTL_DAYS)
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO tokens (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token, user_id, expires_at.isoformat(), now.isoformat()),
        )
    return token


def _resolve_user(authorization: str | None, allow_guest: bool = True) -> dict[str, Any]:
    if not authorization:
        if allow_guest:
            return {"id": 0, "email": "guest@local"}
        raise HTTPException(status_code=401, detail="Missing authorization token.")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization must be Bearer token.")
    token = parts[1].strip()

    with db_conn() as conn:
        row = conn.execute(
            """
            SELECT u.id, u.email, t.expires_at
            FROM tokens t
            JOIN users u ON u.id = t.user_id
            WHERE t.token = ?
            """,
            (token,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=401, detail="Invalid token.")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Token expired.")
    return {"id": int(row["id"]), "email": row["email"]}


def _list_documents() -> list[dict[str, Any]]:
    ensure_data_dirs()
    docs: list[dict[str, Any]] = []
    for path in sorted(RAW_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = path.stat()
        docs.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return docs


def get_engine(refresh: bool = False) -> RAGEngine:
    global _engine
    if _engine is None or refresh:
        _engine = RAGEngine()
    return _engine


def _ensure_chat(session_id: str, user_id: int, title: str | None = None) -> None:
    now = _now_iso()
    with db_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM chats WHERE session_id = ? AND user_id = ?", (session_id, user_id)
        ).fetchone()
        if exists is None:
            chat_title = title or "New chat"
            conn.execute(
                "INSERT INTO chats (session_id, user_id, title, archived, created_at, last_updated) VALUES (?, ?, ?, 0, ?, ?)",
                (session_id, user_id, chat_title, now, now),
            )


def _append_message(
    *,
    session_id: str,
    user_id: int,
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    latency_ms: float,
    top_k: int,
    retrieval_profile: str,
    message_id: str,
    created_at: str,
) -> None:
    with db_conn() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                message_id, session_id, user_id, question, answer, citations_json,
                latency_ms, top_k, retrieval_profile, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                user_id,
                question,
                answer,
                json.dumps(citations, ensure_ascii=False),
                latency_ms,
                top_k,
                retrieval_profile,
                created_at,
            ),
        )
        title_row = conn.execute(
            "SELECT title FROM chats WHERE session_id = ? AND user_id = ?", (session_id, user_id)
        ).fetchone()
        if title_row and title_row["title"] == "New chat":
            title = question.strip()[:80] or "New chat"
            conn.execute(
                "UPDATE chats SET title = ?, last_updated = ? WHERE session_id = ? AND user_id = ?",
                (title, created_at, session_id, user_id),
            )
        else:
            conn.execute(
                "UPDATE chats SET last_updated = ? WHERE session_id = ? AND user_id = ?",
                (created_at, session_id, user_id),
            )


def _chat_messages(session_id: str, user_id: int, limit: int = 500) -> list[dict[str, Any]]:
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT message_id, session_id, question, answer, citations_json, latency_ms, top_k,
                   retrieval_profile, created_at
            FROM messages
            WHERE session_id = ? AND user_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (session_id, user_id, limit),
        ).fetchall()
    return [
        {
            "message_id": row["message_id"],
            "session_id": row["session_id"],
            "question": row["question"],
            "answer": row["answer"],
            "citations": json.loads(row["citations_json"]),
            "latency_ms": float(row["latency_ms"]),
            "top_k": int(row["top_k"]),
            "retrieval_profile": row["retrieval_profile"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


ensure_data_dirs()
STATIC_DIR.mkdir(parents=True, exist_ok=True)
init_db()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def home() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="Frontend not found. Missing src/static/index.html")
    return FileResponse(page)


@app.get("/health")
def health() -> dict[str, Any]:
    cuda = False
    device_name = None
    try:
        import torch

        cuda = torch.cuda.is_available()
        if cuda:
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {"status": "ok", "cuda_available": cuda, "device": device_name}


@app.post("/auth/register", response_model=AuthResponse)
def register(payload: AuthRequest) -> AuthResponse:
    email = payload.email.strip().lower()
    with db_conn() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing is not None:
            raise HTTPException(status_code=409, detail="Email already registered.")
        now = _now_iso()
        pw_hash = _password_hash(payload.password)
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, pw_hash, now),
        )
        user_id = int(cur.lastrowid)
    token = _create_token(user_id)
    return AuthResponse(token=token, user=AuthUser(id=user_id, email=email))


@app.post("/auth/login", response_model=AuthResponse)
def login(payload: AuthRequest) -> AuthResponse:
    email = payload.email.strip().lower()
    with db_conn() as conn:
        row = conn.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
    if row is None or not _verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = _create_token(int(row["id"]))
    return AuthResponse(token=token, user=AuthUser(id=int(row["id"]), email=row["email"]))


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if not authorization:
        return {"status": "ok"}
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        token = parts[1].strip()
        with db_conn() as conn:
            conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
    return {"status": "ok"}


@app.get("/auth/me", response_model=AuthUser)
def me(authorization: str | None = Header(default=None)) -> AuthUser:
    user = _resolve_user(authorization, allow_guest=False)
    return AuthUser(id=user["id"], email=user["email"])


@app.get("/documents")
def documents() -> dict[str, Any]:
    return {"documents": _list_documents()}


@app.get("/index-info")
def index_info(limit_sources: int = Query(default=25, ge=1, le=200)) -> dict[str, Any]:
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        return {"indexed": False, "num_chunks": 0, "sources": []}
    source_counts: dict[str, int] = {}
    total = 0
    for row in read_jsonl(chunks_path):
        total += 1
        src = row.get("source_path", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    ordered = sorted(source_counts.items(), key=lambda item: item[1], reverse=True)
    sources = [{"source": src, "chunks": count} for src, count in ordered[:limit_sources]]
    return {"indexed": True, "num_chunks": total, "sources": sources}


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
    ensure_data_dirs()
    uploaded: list[str] = []
    skipped: list[str] = []
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    for file in files:
        try:
            safe_name = _safe_filename(file.filename or "")
            suffix = Path(safe_name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                skipped.append(f"{file.filename} (unsupported)")
                continue
            content = await file.read()
            if not content:
                skipped.append(f"{safe_name} (empty)")
                continue
            target = RAW_DIR / safe_name
            with target.open("wb") as out:
                out.write(content)
            uploaded.append(safe_name)
        except Exception as exc:
            skipped.append(f"{file.filename} ({exc})")

    if not uploaded and skipped:
        raise HTTPException(status_code=400, detail={"uploaded": uploaded, "skipped": skipped})
    return UploadResponse(status="success", uploaded=uploaded, skipped=skipped)


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    try:
        artifacts = ingest_documents(
            rebuild=payload.rebuild,
            chunk_size_chars=payload.chunk_size_chars,
            overlap_chars=payload.overlap_chars,
        )
        get_engine(refresh=True)
        return IngestResponse(
            status="success",
            num_chunks=artifacts.num_chunks,
            chunks_path=str(artifacts.chunks_path),
            index_path=str(artifacts.index_path),
            metadata_path=str(artifacts.metadata_path),
        )
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    engine = get_engine()
    try:
        result = engine.ask(
            payload.question, top_k=payload.top_k, retrieval_profile=payload.retrieval_profile
        )
        return AskResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, authorization: str | None = Header(default=None)) -> ChatResponse:
    user = _resolve_user(authorization, allow_guest=True)
    session_id = payload.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    created_at = _now_iso()

    _ensure_chat(session_id, user_id=user["id"], title="New chat")

    result = ask(
        AskRequest(
            question=payload.question,
            top_k=payload.top_k,
            retrieval_profile=payload.retrieval_profile,
        )
    )

    _append_message(
        session_id=session_id,
        user_id=user["id"],
        question=payload.question,
        answer=result.answer,
        citations=result.citations,
        latency_ms=result.latency_ms,
        top_k=payload.top_k,
        retrieval_profile=result.retrieval_profile,
        message_id=message_id,
        created_at=created_at,
    )

    return ChatResponse(
        session_id=session_id,
        message_id=message_id,
        question=payload.question,
        answer=result.answer,
        citations=result.citations,
        retrieval_profile=result.retrieval_profile,
        latency_ms=result.latency_ms,
        latency_breakdown_ms=result.latency_breakdown_ms,
        created_at=created_at,
    )


@app.get("/history")
def history(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    user = _resolve_user(authorization, allow_guest=True)
    with db_conn() as conn:
        if session_id:
            rows = conn.execute(
                """
                SELECT message_id, session_id, question, answer, citations_json, latency_ms, top_k,
                       retrieval_profile, created_at
                FROM messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (user["id"], session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT message_id, session_id, question, answer, citations_json, latency_ms, top_k,
                       retrieval_profile, created_at
                FROM messages
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user["id"], limit),
            ).fetchall()

    messages = [
        {
            "message_id": row["message_id"],
            "session_id": row["session_id"],
            "question": row["question"],
            "answer": row["answer"],
            "citations": json.loads(row["citations_json"]),
            "latency_ms": float(row["latency_ms"]),
            "top_k": int(row["top_k"]),
            "retrieval_profile": row["retrieval_profile"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return {"messages": messages}


@app.get("/history/sessions")
def history_sessions(
    limit: int = Query(default=100, ge=1, le=500),
    authorization: str | None = Header(default=None),
) -> dict[str, list[ChatSummary]]:
    user = _resolve_user(authorization, allow_guest=True)
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.session_id, c.title, c.archived, c.created_at, c.last_updated,
                   COUNT(m.message_id) AS turns
            FROM chats c
            LEFT JOIN messages m ON m.session_id = c.session_id AND m.user_id = c.user_id
            WHERE c.user_id = ?
            GROUP BY c.session_id, c.title, c.archived, c.created_at, c.last_updated
            ORDER BY c.last_updated DESC
            LIMIT ?
            """,
            (user["id"], limit),
        ).fetchall()

    sessions = [
        ChatSummary(
            session_id=row["session_id"],
            title=row["title"],
            turns=int(row["turns"]),
            archived=bool(row["archived"]),
            created_at=row["created_at"],
            last_updated=row["last_updated"],
        )
        for row in rows
    ]
    return {"sessions": sessions}


@app.get("/chats")
def list_chats(
    archived: bool = Query(default=False),
    authorization: str | None = Header(default=None),
) -> dict[str, list[ChatSummary]]:
    user = _resolve_user(authorization, allow_guest=False)
    with db_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.session_id, c.title, c.archived, c.created_at, c.last_updated,
                   COUNT(m.message_id) AS turns
            FROM chats c
            LEFT JOIN messages m ON m.session_id = c.session_id AND m.user_id = c.user_id
            WHERE c.user_id = ? AND c.archived = ?
            GROUP BY c.session_id, c.title, c.archived, c.created_at, c.last_updated
            ORDER BY c.last_updated DESC
            """,
            (user["id"], int(archived)),
        ).fetchall()
    return {
        "chats": [
            ChatSummary(
                session_id=row["session_id"],
                title=row["title"],
                turns=int(row["turns"]),
                archived=bool(row["archived"]),
                created_at=row["created_at"],
                last_updated=row["last_updated"],
            )
            for row in rows
        ]
    }


@app.post("/chats")
def create_chat(payload: ChatCreateRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
    user = _resolve_user(authorization, allow_guest=False)
    session_id = str(uuid.uuid4())
    _ensure_chat(session_id, user_id=user["id"], title=payload.title or "New chat")
    return {"session_id": session_id}


@app.get("/chats/{session_id}/messages")
def chat_messages(session_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    user = _resolve_user(authorization, allow_guest=False)
    return {"messages": _chat_messages(session_id=session_id, user_id=user["id"], limit=500)}


@app.patch("/chats/{session_id}")
def update_chat(
    session_id: str,
    payload: ChatUpdateRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    user = _resolve_user(authorization, allow_guest=False)
    with db_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM chats WHERE session_id = ? AND user_id = ?", (session_id, user["id"])
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Chat not found.")

        updates: list[str] = []
        params: list[Any] = []
        if payload.title is not None:
            updates.append("title = ?")
            params.append(payload.title.strip()[:120] or "New chat")
        if payload.archived is not None:
            updates.append("archived = ?")
            params.append(int(payload.archived))
        updates.append("last_updated = ?")
        params.append(_now_iso())
        params.extend([session_id, user["id"]])
        conn.execute(
            f"UPDATE chats SET {', '.join(updates)} WHERE session_id = ? AND user_id = ?",
            params,
        )
    return {"status": "ok"}


@app.delete("/chats/{session_id}")
def delete_chat(session_id: str, authorization: str | None = Header(default=None)) -> dict[str, str]:
    user = _resolve_user(authorization, allow_guest=False)
    with db_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ? AND user_id = ?", (session_id, user["id"]))
        conn.execute("DELETE FROM chats WHERE session_id = ? AND user_id = ?", (session_id, user["id"]))
    return {"status": "ok"}
