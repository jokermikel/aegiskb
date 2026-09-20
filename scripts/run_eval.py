"""Run the checked-in labeled retrieval dataset without starting an HTTP server.

This command initializes the configured database, seeds only an empty demo
database, writes one durable evaluation record, and prints/saves its summary.
Use the API endpoint instead when evaluating a running deployed instance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# `python scripts/run_eval.py` puts scripts/ (rather than repository root) on
# sys.path, so make the project package importable without requiring install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import SessionLocal, initialize_database
from app.evaluation import run_retrieval_evaluation
from app.main import seed_demo_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AegisKB labeled retrieval evaluation")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--report", type=Path, default=Path("output/evaluation/latest.json"))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("top-k 必须为正整数")
    return args


def main() -> None:
    args = parse_args()
    initialize_database()
    with SessionLocal() as db:
        seed_demo_data(db)
        result = run_retrieval_evaluation(db, top_k=args.top_k)
    # Per-case titles and snippets remain in the admin-only database record.
    # The portable report keeps just aggregate metrics suitable for comparison.
    report = {key: value for key, value in result.items() if key != "results"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"报告已写入：{args.report}")


if __name__ == "__main__":
    main()
