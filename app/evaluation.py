"""Offline, labeled retrieval evaluation with durable Recall@K records.

Only cases that declare ``relevant_titles`` contribute to Recall@K, HitRate@K
and MRR. Safety and approval cases may stay in the same JSONL dataset for
broader regression coverage, but do not distort the retrieval denominator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import RetrievalEvaluationRun, User
from .retrieval import RetrievalMetrics, search_knowledge

EVAL_DATASET = Path(__file__).resolve().parents[1] / "data" / "evals.jsonl"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    user_email: str
    query: str
    relevant_titles: tuple[str, ...]


def load_cases(path: Path = EVAL_DATASET) -> tuple[list[EvalCase], str]:
    """Load the versioned JSONL fixture and return cases plus its content hash."""
    raw = path.read_bytes()
    cases: list[EvalCase] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            case_id = str(data["id"])
            user_email = str(data["user"])
            query = str(data["input"])
            relevant_titles = tuple(str(title) for title in data.get("relevant_titles", []))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path.name} 第 {line_number} 行格式无效") from exc
        if not case_id or not user_email or not query:
            raise ValueError(f"{path.name} 第 {line_number} 行缺少 id、user 或 input")
        cases.append(EvalCase(case_id, user_email, query, relevant_titles))
    if not cases:
        raise ValueError(f"{path.name} 没有评测样本")
    return cases, hashlib.sha256(raw).hexdigest()


def _round(value: float) -> float:
    return round(value, 4)


def run_retrieval_evaluation(db: Session, *, top_k: int = 4, dataset: Path = EVAL_DATASET) -> dict[str, Any]:
    """Evaluate current retrieval settings and persist a comparable run result."""
    cases, dataset_hash = load_cases(dataset)
    details: list[dict[str, Any]] = []
    recall_sum = hit_sum = mrr_sum = 0.0
    evaluated = 0
    last_metrics = RetrievalMetrics()

    for case in cases:
        user = db.scalar(select(User).where(User.email == case.user_email))
        if user is None:
            raise ValueError(f"评测样本 {case.case_id} 的用户不存在：{case.user_email}")
        metrics = RetrievalMetrics()
        hits = search_knowledge(db, user, case.query, limit=top_k, metrics=metrics)
        last_metrics = metrics
        titles = [hit.title for hit in hits]
        expected = set(case.relevant_titles)
        matched = expected.intersection(titles)
        rank = next((index for index, title in enumerate(titles, start=1) if title in expected), None)

        case_result: dict[str, Any] = {
            "id": case.case_id,
            "relevant_titles": list(case.relevant_titles),
            "retrieved_titles": titles,
            "metrics": metrics.as_dict(),
        }
        if expected:
            recall = len(matched) / len(expected)
            reciprocal_rank = 0.0 if rank is None else 1.0 / rank
            evaluated += 1
            recall_sum += recall
            hit_sum += 1.0 if matched else 0.0
            mrr_sum += reciprocal_rank
            case_result.update(
                {
                    "recall_at_k": _round(recall),
                    "hit_at_k": bool(matched),
                    "passed_at_k": recall == 1.0,
                    "reciprocal_rank": _round(reciprocal_rank),
                }
            )
        else:
            case_result["excluded_from_retrieval_metrics"] = True
        details.append(case_result)

    if not evaluated:
        raise ValueError("评测集没有带 relevant_titles 的检索样本")

    run = RetrievalEvaluationRun(
        dataset_hash=dataset_hash,
        top_k=top_k,
        total_cases=len(cases),
        evaluated_cases=evaluated,
        recall_at_k=recall_sum / evaluated,
        hit_rate_at_k=hit_sum / evaluated,
        mrr=mrr_sum / evaluated,
        backend=last_metrics.backend,
        embedder=last_metrics.embedder,
        dim=last_metrics.dim,
        results=details,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return evaluation_run_as_dict(run, include_details=True)


def evaluation_run_as_dict(run: RetrievalEvaluationRun, *, include_details: bool = False) -> dict[str, Any]:
    eligible = [item for item in run.results if item.get("relevant_titles")]
    passed = sum(1 for item in eligible if item.get("passed_at_k", item.get("recall_at_k") == 1.0))
    result = {
        "id": run.id,
        "dataset_hash": run.dataset_hash,
        "top_k": run.top_k,
        "total_cases": run.total_cases,
        "evaluated_cases": run.evaluated_cases,
        "recall_at_k": _round(run.recall_at_k),
        "hit_rate_at_k": _round(run.hit_rate_at_k),
        # Unlike HitRate@K (at least one matching document), this only passes
        # a case when every annotated document was retrieved in Top-K.
        "pass_rate_at_k": _round(passed / len(eligible)) if eligible else 0.0,
        "mrr": _round(run.mrr),
        "backend": run.backend,
        "embedder": run.embedder,
        "dim": run.dim,
        "created_at": run.created_at,
    }
    if include_details:
        result["results"] = run.results
    return result


def list_evaluation_runs(db: Session, *, limit: int = 20) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(RetrievalEvaluationRun).order_by(RetrievalEvaluationRun.created_at.desc()).limit(min(max(limit, 1), 100))
    ).all()
    return [evaluation_run_as_dict(row) for row in rows]
