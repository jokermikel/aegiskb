"""Small HTTP-level concurrency benchmark for the running AegisKB API.

Example (start the API separately):

    python scripts/benchmark.py --requests 40 --concurrency 2

For meaningful multi-worker/high-concurrency numbers, point it at the Docker
PostgreSQL deployment. SQLite serializes writes to audit logs, so its numbers
are intended only as a local smoke benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import httpx


def percentile(values: list[float], percent: float) -> float:
    """Linear-interpolated percentile with no NumPy dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def login(client: httpx.AsyncClient, email: str) -> str:
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": "demo-password"})
    response.raise_for_status()
    return response.json()["access_token"]


async def chat_once(client: httpx.AsyncClient, headers: dict[str, str], message: str) -> tuple[float, int]:
    started = time.perf_counter()
    response = await client.post("/api/v1/agent/chat", headers=headers, json={"message": message})
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, response.status_code


async def run(args: argparse.Namespace) -> dict[str, Any]:
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=timeout) as client:
        finance_token, admin_token = await asyncio.gather(
            login(client, "finance@example.com"), login(client, "admin@example.com")
        )
        finance_headers = {"Authorization": f"Bearer {finance_token}"}
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        evaluation = await client.post("/api/v1/retrieval/evaluations", headers=admin_headers, json={"top_k": args.top_k})
        evaluation.raise_for_status()
        evaluation_result = evaluation.json()

        # Build embeddings and the index before latency sampling.
        for _ in range(args.warmup):
            _, status = await chat_once(client, finance_headers, args.message)
            if status != 200:
                raise RuntimeError(f"预热请求失败，HTTP {status}")

        semaphore = asyncio.Semaphore(args.concurrency)

        async def measured_request() -> tuple[float, int]:
            async with semaphore:
                return await chat_once(client, finance_headers, args.message)

        sample_started = time.perf_counter()
        outcomes = await asyncio.gather(*(measured_request() for _ in range(args.requests)))
        sample_elapsed_s = time.perf_counter() - sample_started

    latencies = [elapsed for elapsed, status in outcomes if status == 200]
    failures = [{"status": status, "latency_ms": round(elapsed, 2)} for elapsed, status in outcomes if status != 200]
    if not latencies:
        raise RuntimeError("所有压测请求均失败，未生成延迟指标")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "workload": {
            "message": args.message,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "top_k": args.top_k,
        },
        "latency_ms": {
            "count": len(latencies),
            "min": round(min(latencies), 2),
            "mean": round(fmean(latencies), 2),
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "p99": round(percentile(latencies, 99), 2),
            "max": round(max(latencies), 2),
        },
        "throughput_rps": round(len(latencies) / sample_elapsed_s, 2),
        "error_rate": round(len(failures) / len(outcomes), 4),
        "failures": failures[:20],
        "evaluation": {
            key: evaluation_result[key]
            for key in ("id", "dataset_hash", "top_k", "evaluated_cases", "recall_at_k", "hit_rate_at_k", "pass_rate_at_k", "mrr", "backend", "embedder")
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AegisKB httpx concurrency benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--message", default="国内出差住宿报销标准是多少？")
    parser.add_argument("--report", type=Path, default=Path("output/benchmark/latest.json"))
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0 or args.top_k < 1:
        parser.error("requests、concurrency、top-k 必须为正数，warmup 不能为负数")
    return args


def main() -> None:
    args = parse_args()
    report = asyncio.run(run(args))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入：{args.report}")


if __name__ == "__main__":
    main()
