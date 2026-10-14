"""按保留期清理审计日志。

审计表只增不减，长期运行需要定期清理。清理是不可逆操作，因此**默认只统计不删除**，
必须显式加 `--apply` 才会真正执行。

用法：
    python scripts/prune_audit_logs.py --days 180            # 预演，只报告数量
    python scripts/prune_audit_logs.py --days 180 --apply    # 真正删除
"""
import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# `python scripts/xxx.py` 会把 scripts/ 放进 sys.path，这里补上项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import delete, func, select

from app.db import AuditLog, SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清理超过保留期的审计日志")
    parser.add_argument("--days", type=int, required=True, help="保留最近多少天")
    parser.add_argument("--apply", action="store_true", help="真正执行删除（默认只预演）")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days 必须为正整数")
    return args


def main() -> int:
    args = parse_args()
    cutoff = datetime.now(UTC) - timedelta(days=args.days)

    with SessionLocal() as db:
        total = db.scalar(select(func.count(AuditLog.id))) or 0
        expired = db.scalar(select(func.count(AuditLog.id)).where(AuditLog.created_at < cutoff)) or 0
        print(f"审计日志总数 {total} 条，早于 {cutoff.isoformat()} 的有 {expired} 条")

        if not args.apply:
            print("预演模式：未删除任何记录。确认无误后加 --apply 执行。")
            return 0
        if expired == 0:
            print("没有需要清理的记录。")
            return 0

        db.execute(delete(AuditLog).where(AuditLog.created_at < cutoff))
        db.commit()
        remaining = db.scalar(select(func.count(AuditLog.id))) or 0
        print(f"已清理 {expired} 条，剩余 {remaining} 条。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
