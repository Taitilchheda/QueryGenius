from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from .utils import (
    BASE_DIR,
    INDEX_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ChunkRecord,
    chunk_text,
    ensure_data_dirs,
    list_source_files,
    load_document,
    save_json,
    write_jsonl,
)

logger = logging.getLogger(__name__)

if load_dotenv is not None:
    load_dotenv()

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
INDEX_PATH = INDEX_DIR / "faiss.index"
INDEX_META_PATH = INDEX_DIR / "metadata.json"


def _import_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError(
            "FAISS is required for indexing. Install dependencies from requirements.txt."
        ) from exc
    return faiss


@dataclass
class IngestArtifacts:
    chunks_path: Path
    index_path: Path
    metadata_path: Path
    num_chunks: int


class Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "QG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = os.getenv("QG_EMBEDDING_DEVICE")
        self._model = None
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._lazy_load_model()
        assert self._dim is not None
        return self._dim

    def _lazy_load_model(self) -> None:
        if self._model is not None:
            return
        use_hash = os.getenv("QG_USE_HASH_EMBEDDINGS", "0") == "1"
        if use_hash:
            self._model = "hash"
            self._dim = 384
            return
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=resolved_device)
            self._dim = int(self._model.get_sentence_embedding_dimension())
            logger.info("Loaded embedding model %s on %s", self.model_name, resolved_device)
        except Exception as exc:
            logger.warning(
                "Failed to load SentenceTransformer. Falling back to hash embeddings. Error: %s",
                exc,
            )
            self._model = "hash"
            self._dim = 384

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._lazy_load_model()
        if self._model == "hash":
            return self._hash_encode(texts, self.dimension)
        arr = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return arr.astype("float32")

    @staticmethod
    def _hash_encode(texts: Sequence[str], dim: int = 384) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype="float32")
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue
            for token in tokens:
                idx = hash(token) % dim
                vectors[i, idx] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


def build_chunks(
    raw_dir: Path = RAW_DIR, chunk_size_chars: int = 1500, overlap_chars: int = 200
) -> list[ChunkRecord]:
    files = list_source_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No supported files found in {raw_dir}. Add .txt/.md/.pdf files before ingest."
        )

    all_chunks: list[ChunkRecord] = []
    for file_path in files:
        raw_text = load_document(file_path)
        spans = chunk_text(raw_text, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)
        doc_id = file_path.stem
        try:
            relative_path = str(file_path.relative_to(BASE_DIR)).replace("\\", "/")
        except ValueError:
            relative_path = str(file_path).replace("\\", "/")
        for chunk_id, (start_char, end_char, text) in enumerate(spans):
            all_chunks.append(
                ChunkRecord(
                    doc_id=doc_id,
                    source_path=relative_path,
                    chunk_id=chunk_id,
                    start_char=start_char,
                    end_char=end_char,
                    text=text,
                )
            )
    if not all_chunks:
        raise ValueError("Documents were loaded but no chunks were produced.")
    return all_chunks


def _build_faiss_index(embeddings: np.ndarray) -> Any:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")
    faiss = _import_faiss()
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def ingest_documents(
    raw_dir: Path = RAW_DIR,
    rebuild: bool = True,
    chunk_size_chars: int = 1500,
    overlap_chars: int = 200,
    embedding_model: str | None = None,
    chunks_path: Path = CHUNKS_PATH,
    index_path: Path = INDEX_PATH,
    index_meta_path: Path = INDEX_META_PATH,
) -> IngestArtifacts:
    ensure_data_dirs()
    if not rebuild and index_path.exists() and index_meta_path.exists() and chunks_path.exists():
        metadata = index_meta_path.read_text(encoding="utf-8")
        logger.info("Existing index found. Skipping rebuild. Metadata: %s", metadata[:200])
        rows = sum(1 for _ in chunks_path.open("r", encoding="utf-8"))
        return IngestArtifacts(chunks_path, index_path, index_meta_path, rows)

    chunks = build_chunks(raw_dir=raw_dir, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)
    write_jsonl(chunks_path, [chunk.model_dump() for chunk in chunks])

    resolved_embedding_model = embedding_model or os.getenv(
        "QG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedder = Embedder(model_name=resolved_embedding_model)
    texts = [chunk.text for chunk in chunks]
    t0 = time.perf_counter()
    embeddings = embedder.encode(texts)
    embed_ms = (time.perf_counter() - t0) * 1000.0

    index = _build_faiss_index(embeddings.copy())
    faiss = _import_faiss()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    save_json(
        index_meta_path,
        {
            "embedding_model": resolved_embedding_model,
            "embedding_dim": int(embedder.dimension),
            "num_chunks": len(chunks),
            "chunk_size_chars": chunk_size_chars,
            "overlap_chars": overlap_chars,
            "embedding_time_ms": round(embed_ms, 2),
            "raw_dir": str(raw_dir),
            "chunks_path": str(chunks_path),
        },
    )
    logger.info("Ingestion complete. %d chunks indexed.", len(chunks))
    return IngestArtifacts(chunks_path, index_path, index_meta_path, len(chunks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunks and FAISS index for QueryGenius.")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--rebuild", dest="rebuild", action="store_true", help="Force index rebuild.")
    parser.add_argument("--no-rebuild", dest="rebuild", action="store_false", help="Reuse existing index if present.")
    parser.set_defaults(rebuild=True)
    parser.add_argument("--chunk-size-chars", type=int, default=1500)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    artifacts = ingest_documents(
        raw_dir=Path(args.raw_dir),
        rebuild=args.rebuild,
        chunk_size_chars=args.chunk_size_chars,
        overlap_chars=args.overlap_chars,
        embedding_model=args.embedding_model,
    )
    print(
        f"Ingestion finished. chunks={artifacts.num_chunks} "
        f"index='{artifacts.index_path}' metadata='{artifacts.metadata_path}'"
    )


if __name__ == "__main__":
    main()
