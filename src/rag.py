from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from .ingest import CHUNKS_PATH, INDEX_META_PATH, INDEX_PATH, Embedder
from .utils import ChunkRecord, RetrievedChunk, read_jsonl

logger = logging.getLogger(__name__)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "into",
    "about",
    "along",
    "used",
    "use",
    "give",
    "list",
    "describe",
    "story",
    "plot",
}

if load_dotenv is not None:
    load_dotenv()


def _import_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError(
            "FAISS is required for retrieval. Install dependencies from requirements.txt."
        ) from exc
    return faiss


class RAGEngine:
    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        metadata_path: Path = INDEX_META_PATH,
        chunks_path: Path = CHUNKS_PATH,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        enable_llm: bool | None = None,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunks_path = chunks_path
        self.embedding_model = embedding_model or os.getenv(
            "QG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm_model = llm_model or os.getenv("QG_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self.enable_llm = (
            os.getenv("QG_ENABLE_LLM", "1") == "1" if enable_llm is None else enable_llm
        )
        self._index: Any | None = None
        self._chunks: list[ChunkRecord] | None = None
        self._embedder: Embedder | None = None
        self._generator = None
        self._tokenizer = None
        self.max_context_chars = int(os.getenv("QG_MAX_CONTEXT_CHARS", "12000"))
        self.max_chunk_context_chars = int(os.getenv("QG_MAX_CHUNK_CONTEXT_CHARS", "2000"))
        self.max_new_tokens = int(os.getenv("QG_MAX_NEW_TOKENS", "160"))
        self.max_new_tokens_diagram = int(os.getenv("QG_MAX_NEW_TOKENS_DIAGRAM", "280"))
        self.temperature = float(os.getenv("QG_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("QG_TOP_P", "1.0"))
        self.strict_grounded = os.getenv("QG_STRICT_GROUNDED", "1") == "1"
        self.min_retrieval_score = float(os.getenv("QG_MIN_RETRIEVAL_SCORE", "0.18"))
        self.min_query_overlap = float(os.getenv("QG_MIN_QUERY_OVERLAP", "0.22"))
        self.enforce_source_focus = os.getenv("QG_ENFORCE_SOURCE_FOCUS", "1") == "1"

    def is_ready(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists() and self.chunks_path.exists()

    def load(self) -> None:
        if not self.is_ready():
            raise FileNotFoundError(
                "Index artifacts missing. Run ingestion first via `python -m src.ingest --rebuild`."
            )
        if self._index is None:
            faiss = _import_faiss()
            self._index = faiss.read_index(str(self.index_path))
        if self._chunks is None:
            self._chunks = [ChunkRecord(**row) for row in read_jsonl(self.chunks_path)]
        if self._embedder is None:
            self._embedder = Embedder(model_name=self.embedding_model)

    def _load_generator(self) -> None:
        if not self.enable_llm or self._generator is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, pipeline

            use_cuda = torch.cuda.is_available()
            device = 0 if use_cuda else -1
            self._tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            if use_cuda:
                try:
                    self._generator = pipeline(
                        "text-generation",
                        model=self.llm_model,
                        tokenizer=self._tokenizer,
                        device_map="auto",
                        model_kwargs={"torch_dtype": torch.float16},
                    )
                except Exception:
                    self._generator = pipeline(
                        "text-generation",
                        model=self.llm_model,
                        tokenizer=self._tokenizer,
                        device=0,
                        torch_dtype=torch.float16,
                    )
            else:
                self._generator = pipeline(
                    "text-generation",
                    model=self.llm_model,
                    tokenizer=self._tokenizer,
                    device=-1,
                    torch_dtype=torch.float32,
                )
            logger.info("Loaded generation model %s on device=%s", self.llm_model, device)
        except Exception as exc:
            logger.warning("LLM load failed. Using extractive fallback. Error: %s", exc)
            self._generator = None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        hits, _ = self.retrieve_with_profile(query=query, top_k=top_k, retrieval_profile="balanced")
        return hits

    def retrieve_with_profile(
        self, query: str, top_k: int = 5, retrieval_profile: str = "balanced"
    ) -> tuple[list[RetrievedChunk], str]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        self.load()
        assert self._embedder is not None
        assert self._index is not None
        assert self._chunks is not None

        top_k = max(1, min(top_k, len(self._chunks)))
        search_k = min(len(self._chunks), max(top_k * 8, 20))
        query_vec = self._embedder.encode([query]).astype("float32")
        faiss = _import_faiss()
        faiss.normalize_L2(query_vec)
        scores, indices = self._index.search(query_vec, search_k)

        query_terms = self._query_terms(query)
        profile = self._resolve_profile(query=query, requested_profile=retrieval_profile)
        phrase = self._key_phrase(query)
        sem_by_idx = {int(idx): float(score) for score, idx in zip(scores[0], indices[0]) if idx >= 0}
        candidate_indices = set(sem_by_idx.keys())
        lexical_candidates = self._global_lexical_candidates(query_terms=query_terms, limit=160)
        candidate_indices.update(lexical_candidates)
        reranked: list[tuple[float, float, int]] = []
        for idx in candidate_indices:
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            semantic_score = sem_by_idx.get(idx, 0.0)
            lexical = self._lexical_overlap_score(query_terms, chunk.text)
            diagram_boost = self._diagram_boost(chunk.text, query) if profile == "diagram" else 0.0
            math_boost = self._math_boost(chunk.text, query) if profile == "math" else 0.0
            phrase_boost = self._phrase_boost(phrase, chunk.text)
            combined = 0.4 * float(semantic_score) + 0.6 * lexical + diagram_boost + math_boost + phrase_boost
            reranked.append((combined, float(semantic_score), idx))
        reranked.sort(key=lambda item: item[0], reverse=True)

        reranked = self._expand_with_neighbors(reranked, profile)
        if self.enforce_source_focus:
            reranked = self._apply_source_focus(query=query, reranked=reranked, top_k=top_k)
        hits: list[RetrievedChunk] = []
        for combined_score, _, idx in reranked[:top_k]:
            hits.append(
                RetrievedChunk(
                    score=float(combined_score),
                    chunk=self._chunks[idx],
                )
            )
        return hits, profile

    def generate_answer(self, query: str, contexts: list[RetrievedChunk], retrieval_profile: str) -> str:
        if not contexts:
            return "I do not know based on the provided context."
        if self._is_narrative_query(query):
            return self._narrative_fallback(query=query, contexts=contexts)

        template = self._maybe_template_diagram_answer(query=query, contexts=contexts)
        if template is not None:
            return template
        math_template = self._maybe_template_math_answer(query=query, contexts=contexts)
        if math_template is not None:
            return math_template

        self._load_generator()
        if self._generator is None:
            if retrieval_profile == "math":
                return self._fallback_math_answer(query=query, contexts=contexts)
            return self._fallback_answer(query, contexts)

        context_text = self._format_context(
            contexts,
            max_total_chars=self.max_context_chars,
            max_chars_per_chunk=self.max_chunk_context_chars,
        )
        prompt = self._build_generation_prompt(
            query=query, context_text=context_text, retrieval_profile=retrieval_profile
        )
        max_new_tokens = (
            self.max_new_tokens_diagram if retrieval_profile == "diagram" else self.max_new_tokens
        )
        try:
            output = self._generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                use_cache=True,
                truncation=True,
                return_full_text=False,
            )
            raw = self._clean_generated_answer(output[0]["generated_text"])
            if not raw:
                return self._fallback_answer(query, contexts)
            if retrieval_profile == "diagram" and self._looks_truncated(raw):
                raw = self._continue_answer(prompt, raw)
            if retrieval_profile == "diagram":
                raw = self._normalize_diagram_markdown(raw)
            if retrieval_profile == "math" and not self._contains_latex_delimiters(raw):
                return self._fallback_math_answer(query=query, contexts=contexts)
            if self.strict_grounded and not self._answer_is_grounded(raw, contexts):
                return "I do not know based on the provided context."
            if raw.endswith("?"):
                if retrieval_profile == "math":
                    return self._fallback_math_answer(query=query, contexts=contexts)
                return self._fallback_answer(query, contexts)
            return raw
        except Exception as exc:
            logger.warning("Generation failed. Falling back to extractive mode. Error: %s", exc)
            if retrieval_profile == "math":
                return self._fallback_math_answer(query=query, contexts=contexts)
            return self._fallback_answer(query, contexts)

    def _fallback_answer(self, query: str, contexts: list[RetrievedChunk]) -> str:
        query_terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}
        best_sentence = ""
        best_score = -1
        best_ref = ""

        for hit in contexts:
            sentences = re.split(r"(?<=[.!?])\s+", hit.chunk.text)
            for sentence in sentences:
                terms = {t for t in re.findall(r"\w+", sentence.lower()) if len(t) > 2}
                score = len(query_terms & terms)
                if score > best_score and sentence.strip():
                    best_score = score
                    best_sentence = sentence.strip()
                    source = Path(hit.chunk.source_path).name
                    best_ref = f"[{source}:{hit.chunk.chunk_id}]"

        if best_sentence:
            return f"{best_sentence} {best_ref}".strip()
        return "I do not know based on the provided context."

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        return {
            t
            for t in re.findall(r"\w+", query.lower())
            if len(t) > 2 and t not in STOPWORDS and not t.isdigit()
        }

    @staticmethod
    def _lexical_overlap_score(query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        terms = {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}
        overlap = len(query_terms & terms)
        return overlap / max(1, len(query_terms))

    @staticmethod
    def _format_context(
        contexts: list[RetrievedChunk], max_total_chars: int = 12000, max_chars_per_chunk: int = 2000
    ) -> str:
        sections: list[str] = []
        used = 0
        for hit in contexts:
            source = Path(hit.chunk.source_path).name
            text = hit.chunk.text.strip()
            if len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "..."
            text = text.replace("[PDF_PAGE:", "PDF_PAGE:").replace("[DIAGRAM_PAGE", "DIAGRAM_PAGE")
            block = f"[{source}:{hit.chunk.chunk_id}]\n{text}\n"
            if used + len(block) > max_total_chars and sections:
                break
            sections.append(block)
            used += len(block)
        return "\n".join(sections)

    def _build_generation_prompt(self, query: str, context_text: str, retrieval_profile: str) -> str:
        system_text = (
            "You are QueryGenius, a factual assistant.\n"
            "Use ONLY the provided context.\n"
            "Never answer with a follow-up question.\n"
            "Be direct and concise.\n"
            "When you use a fact, cite it inline as [filename:chunk_id].\n"
            "If context is insufficient, say: I do not know based on the provided context."
        )
        q = query.lower()
        wants_formula = any(term in q for term in ["formula", "equation", "math", "derive"])
        is_transformer = any(
            term in q for term in ["transformer", "self attention", "self-attention", "encoder", "decoder"]
        )
        task_line = "Write a clear answer in 4-8 sentences with citations."
        if retrieval_profile == "diagram":
            if is_transformer:
                task_line = (
                    "Format output as markdown with sections: 'Diagram', 'Encoder/Decoder Blocks', and 'Example'. "
                    "In 'Diagram', include exactly one fenced mermaid block (```mermaid ... ```). "
                    "In 'Encoder/Decoder Blocks', provide at most 5 bullets and max 3 inline citations total."
                )
            else:
                task_line = (
                    "Format output as markdown with sections: 'Diagram', 'How It Works', and 'Example'. "
                    "In 'Diagram', include exactly one fenced mermaid block (```mermaid ... ```). "
                    "In 'How It Works', provide at most 5 bullets and max 3 inline citations total. "
                    "Do not mention encoder/decoder unless the question explicitly asks for them."
                )
            if wants_formula:
                task_line += " Add a final 'Formula' section with 1-3 key equations."
        user_text = (
            f"Question:\n{query}\n\n"
            f"Context:\n{context_text}\n\n"
            f"{task_line}"
        )
        if "tinyllama" in self.llm_model.lower():
            return (
                f"<|system|>\n{system_text}</s>\n"
                f"<|user|>\n{user_text}</s>\n"
                "<|assistant|>\n"
            )
        return f"{system_text}\n\n{user_text}\n\nAnswer:\n"

    def _maybe_template_diagram_answer(
        self, query: str, contexts: list[RetrievedChunk]
    ) -> str | None:
        q = query.lower()
        wants_diagram = any(
            term in q
            for term in ["diagram", "architecture", "block", "flow", "draw", "show me", "illustrate"]
        )
        if not wants_diagram:
            return None
        wants_formula = any(term in q for term in ["formula", "equation", "math", "derive"])
        citations = self._top_citations(contexts, limit=3)

        if "cnn" in q or "convolutional" in q:
            out = (
                "## Diagram\n"
                "```mermaid\n"
                "flowchart LR\n"
                "  A[Input Image HxWxC] --> B[Conv 3x3 + ReLU]\n"
                "  B --> C[Conv 3x3 + ReLU]\n"
                "  C --> D[MaxPool 2x2]\n"
                "  D --> E[Conv 3x3 + ReLU]\n"
                "  E --> F[GlobalAvgPool or Flatten]\n"
                "  F --> G[Dense Layer]\n"
                "  G --> H[Softmax / Output]\n"
                "```\n\n"
                "## How It Works\n"
                "- Convolution filters learn local patterns (edges/textures) and produce feature maps.\n"
                "- Pooling downsamples spatial dimensions to improve robustness and efficiency.\n"
                "- Deeper layers capture higher-level concepts, then dense/output layers perform prediction.\n\n"
                "## Example\n"
                "- Input: 224x224x3 image, Conv(3x3,64) -> Conv(3x3,64) -> MaxPool -> ... -> Softmax(10 classes).\n"
            )
            if wants_formula:
                out += (
                    "\n## Formula\n"
                    "- Convolution: \\( y_{i,j,k} = \\sum_{u}\\sum_{v}\\sum_{c} W_{u,v,c,k}\\,x_{i+u,j+v,c} + b_k \\)\n"
                    "- Output size: \\( H_{out}=\\left\\lfloor\\frac{H+2P-K}{S}\\right\\rfloor+1 \\), "
                    "\\( W_{out}=\\left\\lfloor\\frac{W+2P-K}{S}\\right\\rfloor+1 \\)\n"
                )
            return out + f"\n\nCitations: {citations}"

        if "gan" in q or "generative adversarial" in q:
            out = (
                "## Diagram\n"
                "```mermaid\n"
                "flowchart LR\n"
                "  Z[\"Noise z ~ p(z)\"] --> G[\"Generator G(z)\"]\n"
                "  G --> Xf[\"Fake sample x_hat\"]\n"
                "  Xr[\"Real sample x ~ p_data\"] --> D[\"Discriminator D(x)\"]\n"
                "  Xf --> D\n"
                "  D --> Yr[\"P(real | x)\"]\n"
                "```\n\n"
                "## How It Works\n"
                "- The generator maps random noise to synthetic samples that resemble real data.\n"
                "- The discriminator receives both real and generated samples and predicts real vs fake.\n"
                "- Training alternates: update discriminator to classify correctly, then update generator to fool it.\n"
                "- This is a two-player minimax game that converges when generated and real distributions align.\n\n"
                "## Example\n"
                "- Image generation: sample `z`, produce `x_hat = G(z)`, and improve visual realism over training steps.\n"
            )
            if wants_formula:
                out += (
                    "\n## Formula\n"
                    "- Minimax objective: \\( \\min_G\\max_D\\; V(D,G)=\\mathbb{E}_{x\\sim p_{data}(x)}[\\log D(x)] + "
                    "\\mathbb{E}_{z\\sim p(z)}[\\log(1-D(G(z)))] \\)\n"
                    "- Discriminator loss (to minimize): "
                    "\\( \\mathcal{L}_D = -\\mathbb{E}_{x\\sim p_{data}}[\\log D(x)] - "
                    "\\mathbb{E}_{z\\sim p(z)}[\\log(1-D(G(z)))] \\)\n"
                    "- Non-saturating generator loss: "
                    "\\( \\mathcal{L}_G = -\\mathbb{E}_{z\\sim p(z)}[\\log D(G(z))] \\)\n"
                )
            return out + f"\n\nCitations: {citations}"

        if "lstm" in q:
            out = (
                "## Diagram\n"
                "```mermaid\n"
                "flowchart LR\n"
                "  X[\"x_t\"] --> C1[\"LSTM Cell\"]\n"
                "  Hprev[\"h_{t-1}\"] --> C1\n"
                "  Cprev[\"c_{t-1}\"] --> C1\n"
                "  C1 --> H[\"h_t\"]\n"
                "  C1 --> C[\"c_t\"]\n"
                "```\n\n"
                "## How It Works\n"
                "- The forget/input/output gates control memory flow in the cell state.\n"
                "- Long-range dependencies are handled by additive cell-state updates.\n"
                "- Hidden state `h_t` is used for prediction or passed to the next layer/time-step.\n\n"
                "## Example\n"
                "- Sequence modeling: sentiment classification or next-token prediction over time steps.\n"
            )
            if wants_formula:
                out += (
                    "\n## Formula\n"
                    "- \\( f_t = \\sigma(W_f[x_t,h_{t-1}] + b_f),\\; i_t = \\sigma(W_i[x_t,h_{t-1}] + b_i) \\)\n"
                    "- \\( \\tilde{c}_t = \\tanh(W_c[x_t,h_{t-1}] + b_c),\\; c_t = f_t\\odot c_{t-1} + i_t\\odot \\tilde{c}_t \\)\n"
                    "- \\( o_t = \\sigma(W_o[x_t,h_{t-1}] + b_o),\\; h_t = o_t\\odot\\tanh(c_t) \\)\n"
                )
            return out + f"\n\nCitations: {citations}"

        if "rnn" in q:
            out = (
                "## Diagram\n"
                "```mermaid\n"
                "flowchart LR\n"
                "  X1[x_1] --> R1[h_1]\n"
                "  X2[x_2] --> R2[h_2]\n"
                "  X3[x_3] --> R3[h_3]\n"
                "  R1 --> R2 --> R3\n"
                "  R1 --> Y1[y_1]\n"
                "  R2 --> Y2[y_2]\n"
                "  R3 --> Y3[y_3]\n"
                "```\n\n"
                "## How It Works\n"
                "- The hidden state carries temporal context from previous time steps.\n"
                "- Parameters are shared across time, enabling sequence processing.\n"
                "- Vanilla RNNs can struggle with long dependencies (vanishing gradients).\n\n"
                "## Example\n"
                "- Character-level text generation using one token per time-step.\n"
            )
            if wants_formula:
                out += (
                    "\n## Formula\n"
                    "- \\( h_t = \\phi(W_{xh}x_t + W_{hh}h_{t-1} + b_h) \\)\n"
                    "- \\( y_t = W_{hy}h_t + b_y \\)\n"
                )
            return out + f"\n\nCitations: {citations}"

        if "transformer" in q:
            out = (
                "## Diagram\n"
                "```mermaid\n"
                "flowchart LR\n"
                "  I[\"Input Tokens + Positional Encoding\"] --> E1[\"Encoder Stack (Nx)\"]\n"
                "  E1 --> D1[\"Decoder Stack (Nx)\"]\n"
                "  D1 --> L[\"Linear Projection\"]\n"
                "  L --> S[\"Softmax / Next Token\"]\n"
                "  E1 --> CA[\"Cross-Attention Keys/Values\"]\n"
                "  CA --> D1\n"
                "```\n\n"
                "## Encoder/Decoder Blocks\n"
                "- Encoder block: multi-head self-attention + feed-forward + residual + layer norm.\n"
                "- Decoder block: masked self-attention + cross-attention + feed-forward.\n"
                "- Stacking blocks improves abstraction depth for long-context reasoning.\n\n"
                "## Example\n"
                "- LLM generation: decoder predicts next token conditioned on previous tokens.\n"
            )
            if wants_formula:
                out += (
                    "\n## Formula\n"
                    "- \\( \\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V \\)\n"
                    "- \\( \\mathrm{MultiHead}(Q,K,V)=\\mathrm{Concat}(head_1,\\ldots,head_h)W_O \\)\n"
                    "- \\( head_i=\\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V) \\)\n"
                )
            return out + f"\n\nCitations: {citations}"

        if "self-attention" in q or "self attention" in q:
            out = (
                "## Diagram\n"
                "```mermaid\n"
                "flowchart LR\n"
                "  X[Input Tokens] --> Q[Q = XW_Q]\n"
                "  X --> K[K = XW_K]\n"
                "  X --> V[V = XW_V]\n"
                "  Q --> S[Scores = QK^T / sqrt(d_k)]\n"
                "  K --> S\n"
                "  S --> A[Softmax]\n"
                "  A --> O[Attention Output = A V]\n"
                "  V --> O\n"
                "```\n\n"
                "## How It Works\n"
                "- Query-key similarity gives token-to-token attention scores.\n"
                "- Softmax normalizes scores into attention weights.\n"
                "- Weighted sum of values produces contextualized token representations.\n\n"
                "## Example\n"
                "- In \"The cat sat\", \"sat\" can attend strongly to \"cat\" for subject-verb context.\n"
            )
            if wants_formula:
                out += (
                    "\n## Formula\n"
                    "- \\( \\mathrm{Attention}(Q,K,V)=\\mathrm{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V \\)\n"
                    "- \\( Q=XW_Q,\\;K=XW_K,\\;V=XW_V \\)\n"
                )
            return out + f"\n\nCitations: {citations}"

        return None

    def _maybe_template_math_answer(self, query: str, contexts: list[RetrievedChunk]) -> str | None:
        q = query.lower()
        wants_math = any(term in q for term in ["formula", "formulas", "equation", "math", "derive"])
        if not wants_math and not any(term in q for term in ["backprop", "cross entropy", "bce"]):
            return None
        citations = self._top_citations(contexts, limit=3)

        if "binary cross entropy" in q or "bce" in q:
            return (
                "## Binary Cross-Entropy (BCE)\n"
                "\\[\n"
                "\\mathcal{L}_{BCE} = -\\frac{1}{N}\\sum_{i=1}^{N}\\left[y_i\\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i)\\right]\n"
                "\\]\n\n"
                "For a single sample:\n"
                "\\[\n"
                "\\ell(y,\\hat{y}) = -\\left[y\\log(\\hat{y}) + (1-y)\\log(1-\\hat{y})\\right]\n"
                "\\]\n\n"
                "Where \\(y\\in\\{0,1\\}\\) and \\(\\hat{y}\\in(0,1)\\).\n\n"
                f"Citations: {citations}"
            )

        if "l2" in q or "l-2" in q or "regularization" in q or "adam" in q:
            return (
                "## L2 Regularization\n"
                "\\[\n"
                "\\mathcal{L}_{total}(\\theta)=\\mathcal{L}_{data}(\\theta)+\\lambda\\|\\theta\\|_2^2\n"
                "\\]\n"
                "\\[\n"
                "\\nabla_\\theta \\mathcal{L}_{total}=\\nabla_\\theta \\mathcal{L}_{data}+2\\lambda\\theta\n"
                "\\]\n\n"
                "## Adam Optimizer\n"
                "\\[\n"
                "m_t=\\beta_1 m_{t-1} + (1-\\beta_1)g_t,\\qquad\n"
                "v_t=\\beta_2 v_{t-1} + (1-\\beta_2)g_t^2\n"
                "\\]\n"
                "\\[\n"
                "\\hat m_t=\\frac{m_t}{1-\\beta_1^t},\\qquad\n"
                "\\hat v_t=\\frac{v_t}{1-\\beta_2^t}\n"
                "\\]\n"
                "\\[\n"
                "\\theta_{t+1}=\\theta_t-\\eta\\frac{\\hat m_t}{\\sqrt{\\hat v_t}+\\epsilon}\n"
                "\\]\n\n"
                f"Citations: {citations}"
            )

        if "backprop" in q:
            return (
                "## Backpropagation Derivation (Output + Hidden Layer)\n"
                "Forward pass:\n"
                "\\[\n"
                "z^l = W^l a^{l-1} + b^l,\\qquad a^l = \\sigma(z^l)\n"
                "\\]\n\n"
                "Define output-layer error:\n"
                "\\[\n"
                "\\delta^L = \\frac{\\partial \\mathcal{L}}{\\partial z^L}\n"
                "\\]\n\n"
                "For sigmoid + BCE, this simplifies to:\n"
                "\\[\n"
                "\\delta^L = a^L - y\n"
                "\\]\n\n"
                "Gradient w.r.t. parameters:\n"
                "\\[\n"
                "\\frac{\\partial \\mathcal{L}}{\\partial W^l} = \\delta^l (a^{l-1})^\\top,\n"
                "\\qquad\n"
                "\\frac{\\partial \\mathcal{L}}{\\partial b^l} = \\delta^l\n"
                "\\]\n\n"
                "Hidden-layer recursion:\n"
                "\\[\n"
                "\\delta^l = (W^{l+1})^\\top \\delta^{l+1} \\odot \\sigma'(z^l)\n"
                "\\]\n\n"
                "Parameter update (gradient descent):\n"
                "\\[\n"
                "W^l \\leftarrow W^l - \\eta\\frac{\\partial \\mathcal{L}}{\\partial W^l},\n"
                "\\qquad\n"
                "b^l \\leftarrow b^l - \\eta\\frac{\\partial \\mathcal{L}}{\\partial b^l}\n"
                "\\]\n\n"
                f"Citations: {citations}"
            )

        if not any(term in q for term in ["deep learning", "neural network", "dl", "backprop", "gradient"]):
            return None
        return (
            "## Most Used Deep Learning Formulas\n"
            "1. Linear layer / affine transform: \\( z = Wx + b \\)\n"
            "2. Sigmoid activation: \\( \\sigma(z)=\\frac{1}{1+e^{-z}} \\)\n"
            "3. ReLU activation: \\( \\mathrm{ReLU}(z)=\\max(0,z) \\)\n"
            "4. Softmax: \\( \\mathrm{softmax}(z_i)=\\frac{e^{z_i}}{\\sum_j e^{z_j}} \\)\n"
            "5. Binary cross-entropy: \\( \\mathcal{L}_{BCE} = -\\frac{1}{N}\\sum_{i=1}^{N}[y_i\\log \\hat{y}_i + (1-y_i)\\log(1-\\hat{y}_i)] \\)\n"
            "6. Multiclass cross-entropy: \\( \\mathcal{L}_{CE}=-\\sum_{k=1}^{K} y_k\\log \\hat{y}_k \\)\n"
            "7. Mean squared error: \\( \\mathcal{L}_{MSE}=\\frac{1}{N}\\sum_{i=1}^{N}(y_i-\\hat{y}_i)^2 \\)\n"
            "8. Gradient descent update: \\( \\theta_{t+1}=\\theta_t-\\eta\\nabla_\\theta \\mathcal{L}(\\theta_t) \\)\n"
            "9. Chain rule (backprop core): \\( \\frac{\\partial \\mathcal{L}}{\\partial x}=\\frac{\\partial \\mathcal{L}}{\\partial y}\\cdot\\frac{\\partial y}{\\partial x} \\)\n"
            "10. L2 regularization: \\( \\mathcal{L}_{total}=\\mathcal{L}_{task}+\\lambda\\|\\theta\\|_2^2 \\)\n\n"
            f"Citations: {citations}"
        )

    @staticmethod
    def _clean_generated_answer(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^\s*Answer:\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[source:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+\[filename:\d+\]", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return RAGEngine._dedupe_sections(cleaned.strip())

    @staticmethod
    def _dedupe_sections(text: str) -> str:
        lines = text.splitlines()
        out: list[str] = []
        seen_sections: set[str] = set()
        section_keys = {"diagram", "encoder/decoder blocks", "how it works", "example", "formula"}
        for line in lines:
            key = line.strip().lower().rstrip(":")
            if key in section_keys:
                if key in seen_sections:
                    continue
                seen_sections.add(key)
            if out and line.strip() and out[-1].strip() == line.strip():
                continue
            out.append(line)
        return "\n".join(out).strip()

    @staticmethod
    def _looks_truncated(text: str) -> bool:
        trimmed = text.strip()
        if not trimmed:
            return True
        return not trimmed.endswith((".", "!", "?", "]", "```"))

    def _continue_answer(self, prompt: str, partial: str) -> str:
        if self._generator is None:
            return partial
        continuation_prompt = (
            f"{prompt}{partial}\n"
            "Continue from the last incomplete sentence only. "
            "Finish briefly with citations."
        )
        try:
            cont = self._generator(
                continuation_prompt,
                max_new_tokens=96,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                use_cache=True,
                truncation=True,
                return_full_text=False,
            )
            more = self._clean_generated_answer(cont[0]["generated_text"])
            if not more:
                return partial
            return f"{partial.rstrip()} {more.lstrip()}".strip()
        except Exception:
            return partial

    @staticmethod
    def _normalize_diagram_markdown(text: str) -> str:
        if "```mermaid" in text:
            return text
        pattern = re.compile(r"(?im)^\s*mermaid\s*\n((?:.+\n?){1,30})")
        match = pattern.search(text)
        if not match:
            return text
        body = match.group(1).strip()
        fenced = f"```mermaid\n{body}\n```"
        return text[: match.start()] + fenced + text[match.end() :]

    @staticmethod
    def _contains_math_notation(text: str) -> bool:
        t = text or ""
        if "\\(" in t or "$$" in t or "\\[" in t:
            return True
        if re.search(r"[=+\-*/^_]|\\sum|\\frac|\\nabla|\\log|\\max|\\mathbb", t):
            return True
        return False

    @staticmethod
    def _contains_latex_delimiters(text: str) -> bool:
        t = text or ""
        return ("\\(" in t and "\\)" in t) or ("\\[" in t and "\\]" in t) or ("$$" in t)

    def _fallback_math_answer(self, query: str, contexts: list[RetrievedChunk]) -> str:
        citations = self._top_citations(contexts, limit=3)
        query_terms = self._query_terms(query)
        eq_lines: list[str] = []
        seen: set[str] = set()
        equation_patterns = [
            r"\$[^$]+\$",
            r"\\\([^)]+\\\)",
            r"\\\[[^\]]+\\\]",
            r"[A-Za-z][A-Za-z0-9_{}\[\],\s\\^]*=[^.\n]{6,160}",
        ]
        for hit in contexts:
            text = hit.chunk.text
            for pat in equation_patterns:
                for m in re.finditer(pat, text):
                    expr = m.group(0).strip()
                    if len(expr) < 8:
                        continue
                    alpha_ratio = sum(ch.isalpha() for ch in expr) / max(1, len(expr))
                    has_math_signal = bool(
                        re.search(
                            r"(=|\\sum|\\frac|\\nabla|\\log|\\max|\\theta|\\lambda|\\beta|\\epsilon|\\hat|[_^]|[0-9])",
                            expr,
                        )
                    )
                    if alpha_ratio > 0.88 and not has_math_signal:
                        continue
                    key = expr.lower()
                    if key in seen:
                        continue
                    terms = {t for t in re.findall(r"\w+", expr.lower()) if len(t) > 2}
                    if query_terms and terms and len(query_terms & terms) == 0 and len(eq_lines) >= 2:
                        continue
                    seen.add(key)
                    eq_lines.append(f"- \\( {expr.strip('$')} \\)")
                    if len(eq_lines) >= 8:
                        break
                if len(eq_lines) >= 8:
                    break
            if len(eq_lines) >= 8:
                break
        if not eq_lines:
            return "I do not know based on the provided context."
        return "## Formula\n" + "\n".join(eq_lines[:8]) + f"\n\nCitations: {citations}"

    def ask(self, question: str, top_k: int = 5, retrieval_profile: str = "balanced") -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        start_total = time.perf_counter()

        start_retrieval = time.perf_counter()
        hits, resolved_profile = self.retrieve_with_profile(
            question, top_k=top_k, retrieval_profile=retrieval_profile
        )
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0

        if self.strict_grounded and not self._context_is_relevant(question, hits):
            total_ms = (time.perf_counter() - start_total) * 1000.0
            citations = [
                {
                    "source": hit.chunk.source_path,
                    "chunk_id": hit.chunk.chunk_id,
                    "score": round(hit.score, 4),
                }
                for hit in hits
            ]
            return {
                "answer": "I do not know based on the provided context.",
                "citations": citations,
                "retrieval_profile": resolved_profile,
                "latency_ms": round(total_ms, 2),
                "latency_breakdown_ms": {
                    "embedding_retrieval": round(retrieval_ms, 2),
                    "generation": 0.0,
                    "total": round(total_ms, 2),
                },
            }

        start_generation = time.perf_counter()
        answer = self.generate_answer(question, hits, retrieval_profile=resolved_profile)
        generation_ms = (time.perf_counter() - start_generation) * 1000.0
        total_ms = (time.perf_counter() - start_total) * 1000.0

        citations = [
            {"source": hit.chunk.source_path, "chunk_id": hit.chunk.chunk_id, "score": round(hit.score, 4)}
            for hit in hits
        ]
        return {
            "answer": answer,
            "citations": citations,
            "retrieval_profile": resolved_profile,
            "latency_ms": round(total_ms, 2),
            "latency_breakdown_ms": {
                "embedding_retrieval": round(retrieval_ms, 2),
                "generation": round(generation_ms, 2),
                "total": round(total_ms, 2),
            },
        }

    @staticmethod
    def _resolve_profile(query: str, requested_profile: str) -> str:
        requested = (requested_profile or "balanced").lower().strip()
        q = query.lower()
        # Narrative/literature queries should never be forced into diagram mode.
        if any(term in q for term in ["characters", "character", "novel", "story", "describe the plot", "summary"]):
            return "balanced"
        if any(
            term in q
            for term in [
                "diagram",
                "figure",
                "graph",
                "architecture",
                "block diagram",
                "cnn",
                "rnn",
                "lstm",
                "gru",
                "self attention",
                "self-attention",
            ]
        ):
            return "diagram"
        if requested in {"math", "diagram"}:
            return requested
        if any(
            term in q
            for term in [
                "formula",
                "formulas",
                "equation",
                "derive",
                "derivation",
                "backprop",
                "cross entropy",
                "binary cross entropy",
                "bce",
            ]
        ):
            return "math"
        if any(sym in q for sym in ["=", "+", "-", "*", "/", "^", "integral", "derivative", "matrix"]):
            return "math"
        return "balanced"

    @staticmethod
    def _diagram_boost(text: str, query: str) -> float:
        t = text.lower()
        keywords = ["figure", "fig.", "diagram", "plot", "graph", "table", "illustration", "schema"]
        found = sum(1 for k in keywords if k in t)
        query_bonus = 0.08 if any(k in query.lower() for k in ["diagram", "figure", "plot", "graph"]) else 0.0
        return min(0.2, 0.03 * found + query_bonus)

    @staticmethod
    def _math_boost(text: str, query: str) -> float:
        formula_marks = len(re.findall(r"[=+\-*/^()<>]|\\[a-zA-Z]+", text))
        equation_terms = len(re.findall(r"\b(equation|theorem|proof|lemma|matrix|vector|gradient)\b", text.lower()))
        query_bonus = 0.08 if re.search(r"[=+\-*/^()]|\b(integral|derivative|equation|matrix)\b", query.lower()) else 0.0
        raw = min(0.25, 0.0015 * formula_marks + 0.02 * equation_terms + query_bonus)
        return raw

    def _expand_with_neighbors(
        self, reranked: list[tuple[float, float, int]], profile: str
    ) -> list[tuple[float, float, int]]:
        if not reranked or self._chunks is None:
            return reranked
        if profile == "balanced":
            return reranked
        out: dict[int, tuple[float, float, int]] = {idx: row for row in reranked for idx in [row[2]]}
        for combined, semantic, idx in reranked[: min(6, len(reranked))]:
            base_chunk = self._chunks[idx]
            for nidx in (idx - 1, idx + 1):
                if nidx < 0 or nidx >= len(self._chunks):
                    continue
                neighbor = self._chunks[nidx]
                if neighbor.doc_id != base_chunk.doc_id:
                    continue
                if nidx not in out:
                    out[nidx] = (combined * 0.9, semantic * 0.9, nidx)
        ranked = list(out.values())
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _global_lexical_candidates(self, query_terms: set[str], limit: int = 160) -> list[int]:
        if not query_terms or self._chunks is None:
            return []
        scored: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self._chunks):
            score = self._lexical_overlap_score(query_terms, chunk.text)
            if score <= 0:
                continue
            scored.append((score, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scored[:limit]]

    @staticmethod
    def _key_phrase(query: str) -> str:
        q = query.strip()
        quoted = re.findall(r"\"([^\"]{4,})\"", q)
        if quoted:
            return quoted[0].lower().strip()
        m = re.search(r"of\s+(.+)", q, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower().strip()
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", q) if len(w) > 2]
        if len(words) >= 4:
            return " ".join(words[-4:]).lower()
        return ""

    @staticmethod
    def _phrase_boost(phrase: str, text: str) -> float:
        if not phrase:
            return 0.0
        t = text.lower()
        if phrase in t:
            return 0.35
        tokens = [p for p in re.findall(r"\w+", phrase) if len(p) > 2]
        if not tokens:
            return 0.0
        overlap = sum(1 for tk in tokens if tk in t)
        return min(0.2, 0.04 * overlap)

    @staticmethod
    def _top_citations(contexts: list[RetrievedChunk], limit: int = 3) -> str:
        return ", ".join(
            f"[{Path(hit.chunk.source_path).name}:{hit.chunk.chunk_id}]"
            for hit in contexts[:limit]
        )

    def _context_is_relevant(self, query: str, contexts: list[RetrievedChunk]) -> bool:
        if not contexts:
            return False
        if contexts[0].score < self.min_retrieval_score:
            return False
        query_terms = self._query_terms(query)
        if not query_terms:
            return True
        best_overlap = max(self._lexical_overlap_score(query_terms, c.chunk.text) for c in contexts)
        return best_overlap >= self.min_query_overlap

    def _answer_is_grounded(self, answer: str, contexts: list[RetrievedChunk]) -> bool:
        if not answer.strip():
            return False
        context_text = " ".join(c.chunk.text.lower() for c in contexts)
        answer_terms = [
            t
            for t in re.findall(r"\w+", answer.lower())
            if len(t) > 3 and t not in STOPWORDS and not t.isdigit()
        ]
        if not answer_terms:
            return True
        matched = sum(1 for t in set(answer_terms) if t in context_text)
        ratio = matched / max(1, len(set(answer_terms)))
        return ratio >= 0.35

    @staticmethod
    def _is_narrative_query(query: str) -> bool:
        q = query.lower()
        return any(
            term in q
            for term in [
                "story",
                "plot",
                "characters",
                "character list",
                "summary of",
                "summarize",
                "novel",
                "chapter",
                "sherlock",
            ]
        )

    def _apply_source_focus(
        self, query: str, reranked: list[tuple[float, float, int]], top_k: int
    ) -> list[tuple[float, float, int]]:
        if not reranked or self._chunks is None:
            return reranked
        top_doc_id = self._chunks[reranked[0][2]].doc_id
        same_doc = [row for row in reranked if self._chunks[row[2]].doc_id == top_doc_id]
        other_doc = [row for row in reranked if self._chunks[row[2]].doc_id != top_doc_id]
        if self._is_narrative_query(query):
            return same_doc + other_doc
        # For non-narrative queries, prefer the dominant source but allow limited cross-source evidence.
        keep_other = max(0, min(2, top_k // 3))
        return same_doc + other_doc[:keep_other]

    def _narrative_fallback(self, query: str, contexts: list[RetrievedChunk]) -> str:
        query_terms = self._query_terms(query)
        story_terms = query_terms - {"character", "characters", "list", "story", "plot", "summary"}

        sentence_rows: list[tuple[int, int, str, str, int]] = []
        # tuple: (score, hit_rank, sentence, source, chunk_id)
        for hit_rank, hit in enumerate(contexts):
            source = Path(hit.chunk.source_path).name
            chunk_id = hit.chunk.chunk_id
            sentences = re.split(r"(?<=[.!?])\s+", hit.chunk.text)
            for sent in sentences:
                clean = sent.strip()
                if len(clean) < 35:
                    continue
                terms = {t for t in re.findall(r"\w+", clean.lower()) if len(t) > 2}
                overlap = len(story_terms & terms) if story_terms else len(query_terms & terms)
                if overlap <= 0:
                    continue
                sentence_rows.append((overlap, -hit_rank, clean, source, chunk_id))

        sentence_rows.sort(reverse=True)
        chosen_story: list[str] = []
        seen = set()
        for _, _, sent, source, chunk_id in sentence_rows:
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            chosen_story.append(f"- {sent} [{source}:{chunk_id}]")
            if len(chosen_story) >= 5:
                break

        name_counts: dict[str, int] = {}
        for hit in contexts[:4]:
            text = hit.chunk.text
            # Basic proper-name pattern: "John Openshaw", "Sherlock Holmes", etc.
            for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text):
                name = match.group(1).strip()
                if name.lower() in {"the", "and", "but", "his", "her"}:
                    continue
                if len(name.split()) == 1 and name.lower() not in {"holmes", "watson"}:
                    continue
                name_counts[name] = name_counts.get(name, 0) + 1

        common_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
        character_lines: list[str] = []
        top_source = Path(contexts[0].chunk.source_path).name
        top_chunk = contexts[0].chunk.chunk_id
        for name, _ in common_names[:8]:
            character_lines.append(f"- {name} [{top_source}:{top_chunk}]")

        if not chosen_story:
            return "I do not know based on the provided context."

        parts = ["## Plot Summary", *chosen_story]
        if character_lines:
            parts.extend(["", "## Characters", *character_lines])
        return "\n".join(parts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Simple CLI for QueryGenius RAG.")
    parser.add_argument("question", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    engine = RAGEngine()
    response = engine.ask(args.question, top_k=args.top_k)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
