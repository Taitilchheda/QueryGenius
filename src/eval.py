from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

from .rag import RAGEngine
from .utils import EVAL_DIR, load_json, save_json

if load_dotenv is not None:
    load_dotenv()

EVAL_QUESTIONS_PATH = EVAL_DIR / "eval_questions.json"
REPORT_PATH = EVAL_DIR / "report.json"


def _hit_at_k(citations: list[dict[str, Any]], expected_sources: set[str], k: int) -> int:
    for item in citations[:k]:
        source_name = Path(item["source"]).name
        if source_name in expected_sources:
            return 1
    return 0


def run_evaluation(eval_path: Path = EVAL_QUESTIONS_PATH, top_k: int = 5) -> dict[str, Any]:
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")
    questions = load_json(eval_path)
    if not isinstance(questions, list) or not questions:
        raise ValueError("Eval questions must be a non-empty list.")

    engine = RAGEngine()
    recall1: list[int] = []
    recall3: list[int] = []
    recall5: list[int] = []
    embed_retrieval_lat: list[float] = []
    generation_lat: list[float] = []
    total_lat: list[float] = []
    rows: list[dict[str, Any]] = []

    for item in questions:
        question = item["question"]
        expected_sources = set(item.get("expected_sources", []))
        expected_keywords = [k.lower() for k in item.get("expected_keywords", [])]

        result = engine.ask(question, top_k=top_k)
        citations = result["citations"]
        answer_text = result["answer"].lower()

        recall1.append(_hit_at_k(citations, expected_sources, 1))
        recall3.append(_hit_at_k(citations, expected_sources, 3))
        recall5.append(_hit_at_k(citations, expected_sources, 5))
        embed_retrieval_lat.append(result["latency_breakdown_ms"]["embedding_retrieval"])
        generation_lat.append(result["latency_breakdown_ms"]["generation"])
        total_lat.append(result["latency_breakdown_ms"]["total"])

        keyword_hits = [kw for kw in expected_keywords if kw in answer_text]
        rows.append(
            {
                "question": question,
                "expected_sources": sorted(expected_sources),
                "retrieved_sources_top5": [Path(c["source"]).name for c in citations[:5]],
                "keyword_hits": keyword_hits,
                "latency_breakdown_ms": result["latency_breakdown_ms"],
            }
        )

    report = {
        "num_questions": len(questions),
        "recall_at_1": round(sum(recall1) / len(recall1), 4),
        "recall_at_3": round(sum(recall3) / len(recall3), 4),
        "recall_at_5": round(sum(recall5) / len(recall5), 4),
        "avg_latency_ms": {
            "embedding_retrieval": round(statistics.mean(embed_retrieval_lat), 2),
            "generation": round(statistics.mean(generation_lat), 2),
            "total": round(statistics.mean(total_lat), 2),
        },
        "per_question": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QueryGenius retrieval evaluation.")
    parser.add_argument("--eval-path", type=str, default=str(EVAL_QUESTIONS_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    report = run_evaluation(eval_path=Path(args.eval_path), top_k=args.top_k)
    save_json(REPORT_PATH, report)

    print("=== QueryGenius Evaluation Report ===")
    print(f"Questions: {report['num_questions']}")
    print(f"Recall@1: {report['recall_at_1']}")
    print(f"Recall@3: {report['recall_at_3']}")
    print(f"Recall@5: {report['recall_at_5']}")
    print(f"Avg latency (embedding+retrieval): {report['avg_latency_ms']['embedding_retrieval']} ms")
    print(f"Avg latency (generation): {report['avg_latency_ms']['generation']} ms")
    print(f"Avg latency (total): {report['avg_latency_ms']['total']} ms")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
