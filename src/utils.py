from __future__ import annotations

import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, Field
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class ChunkRecord(BaseModel):
    doc_id: str = Field(..., description="Document identifier based on file stem")
    source_path: str = Field(..., description="Relative source path")
    chunk_id: int
    start_char: int
    end_char: int
    text: str


class RetrievedChunk(BaseModel):
    score: float
    chunk: ChunkRecord


def ensure_data_dirs() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, INDEX_DIR, EVAL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def list_source_files(raw_dir: Path | None = None) -> list[Path]:
    raw_root = raw_dir or RAW_DIR
    files = [
        p
        for p in raw_root.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise ImportError("PDF support requires pypdf. Install dependencies from requirements.txt.") from exc

    reader = PdfReader(str(path))
    pages: list[str] = [page.extract_text() or "" for page in reader.pages]

    enable_ocr = os.getenv("QG_ENABLE_OCR", "1") == "1"
    image_pages: dict[int, int] = {}
    if enable_ocr:
        image_pages = _pdf_image_pages(path)
    if enable_ocr:
        min_chars = int(os.getenv("QG_OCR_MIN_PAGE_CHARS", "80"))
        max_pages = int(os.getenv("QG_OCR_MAX_PAGES", "40"))
        sparse_pages = [idx for idx, txt in enumerate(pages) if len((txt or "").strip()) < min_chars]
        diagram_pages = sorted(image_pages.keys())
        ocr_targets = sorted(set(sparse_pages + diagram_pages))[:max_pages]
        if ocr_targets:
            ocr_text = _ocr_pdf_pages(path, ocr_targets)
            for idx, txt in ocr_text.items():
                if idx < len(pages) and txt.strip():
                    pages[idx] = (pages[idx] + "\n" + txt).strip()

    # Preserve page boundaries and add diagram markers so retrieval can target figure-heavy pages.
    stitched_pages: list[str] = []
    for idx, page_text in enumerate(pages):
        marker = f"[PDF_PAGE:{idx + 1}]"
        if idx in image_pages:
            marker += f" [DIAGRAM_PAGE images={image_pages[idx]}]"
        stitched_pages.append(f"{marker}\n{(page_text or '').strip()}")
    return "\n\n".join(stitched_pages)


def _pdf_image_pages(path: Path) -> dict[int, int]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        return {}
    out: dict[int, int] = {}
    try:
        doc = fitz.open(str(path))
    except Exception:
        return out
    with doc:
        for idx in range(len(doc)):
            try:
                count = len(doc[idx].get_images(full=True))
            except Exception:
                count = 0
            if count > 0:
                out[idx] = count
    return out


def _ocr_pdf_pages(path: Path, page_indices: list[int]) -> dict[int, str]:
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except Exception as exc:
        logger.warning("OCR dependencies missing. Skipping OCR for %s: %s", path.name, exc)
        return {}

    tesseract_cmd = os.getenv("QG_TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    dpi = int(os.getenv("QG_OCR_DPI", "180"))
    zoom = max(dpi / 72.0, 1.0)
    matrix = fitz.Matrix(zoom, zoom)
    lang = os.getenv("QG_OCR_LANG", "eng")

    out: dict[int, str] = {}
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.warning("Failed to open PDF for OCR (%s): %s", path.name, exc)
        return out

    with doc:
        for idx in page_indices:
            if idx < 0 or idx >= len(doc):
                continue
            try:
                page = doc[idx]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img = Image.open(BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang=lang) or ""
                out[idx] = text.strip()
            except Exception as exc:
                logger.warning("OCR failed for %s page %d: %s", path.name, idx + 1, exc)
    return out


def normalize_text(text: str) -> str:
    # Drop invalid surrogate code points that can appear in extracted PDF text.
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str, chunk_size_chars: int = 1500, overlap_chars: int = 200
) -> list[tuple[int, int, str]]:
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars")

    normalized = normalize_text(text)
    if not normalized:
        return []

    chunks: list[tuple[int, int, str]] = []
    start = 0
    step = chunk_size_chars - overlap_chars
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size_chars)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= len(normalized):
            break
        start += step
    return chunks


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
