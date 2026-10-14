"""向量索引：pgvector（Postgres）/ FAISS（本地）/ NumPy（最小依赖兜底）。

三套后端共用同一个 `VectorIndex` 契约，`build_index()` 按当前数据库与已安装的库
自动选型：

| 后端     | 触发条件                                  | 适用场景                         |
| -------- | ----------------------------------------- | -------------------------------- |
| pgvector | 数据源是 PostgreSQL 且 vector 扩展可用    | 生产部署，索引与权限过滤同一事务 |
| faiss    | 装了 `faiss-cpu`（本地 SQLite 场景即可用） | 单机向量检索、演示               |
| numpy    | 以上都不可用                              | 最小依赖环境，语义与 FAISS 一致  |

**安全边界（不要弄反）**：三个后端的 `search()` 只接受"已经通过权限过滤的候选行"。
RBAC 判断发生在 SQL 层或 agent 侧，向量库这一层不做任何权限决策——把向量库当
选型过滤器，越权数据会先被排序、再被丢弃，等于让模型有机会看到不该看的分数。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorRecord:
    """一条待索引的向量。`chunk_id` 是唯一键，`meta` 只放展示需要的字段。"""

    chunk_id: int
    vector: list[float]
    meta: dict[str, Any]


@dataclass
class IndexHit:
    chunk_id: int
    score: float
    meta: dict[str, Any] | None = None


class VectorIndex(Protocol):
    backend: str
    dim: int

    def upsert(self, records: Sequence[VectorRecord]) -> int: ...

    def delete(self, chunk_ids: Sequence[int]) -> int: ...

    def search(self, vector: Sequence[float], allowed_chunk_ids: Sequence[int] | None, limit: int) -> list[IndexHit]: ...

    def rebuild(self, records: Sequence[VectorRecord]) -> None: ...

    def stats(self) -> dict[str, Any]: ...

    def reset(self) -> None: ...


class NumpyIndex:
    """暴力精确检索。归一化向量做内积即余弦相似度，行数在万级以内足够快。"""

    backend = "numpy"

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[int] = []
        self._metas: list[dict] = []
        self._vectors: list[list[float]] = []
        self._matrix = None  # np.ndarray | None

    def _sync_matrix(self) -> None:
        import numpy as np

        self._matrix = np.vstack([np.asarray(v, dtype="float32") for v in self._vectors]) if self._vectors else None

    def _assign(self, ids, metas, vectors) -> None:
        self._ids = list(ids)
        self._metas = list(metas)
        self._vectors = list(vectors)
        self._sync_matrix()

    def _check_dim(self, vector: Sequence[float], what: str) -> None:
        if len(vector) != self.dim:
            raise ValueError(f"{what}维度 {len(vector)} 与索引维度 {self.dim} 不一致")

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        positions = {chunk_id: i for i, chunk_id in enumerate(self._ids)}
        for record in records:
            self._check_dim(record.vector, f"chunk {record.chunk_id} ")
            if record.chunk_id in positions:
                i = positions[record.chunk_id]
                self._vectors[i] = list(record.vector)
                self._metas[i] = dict(record.meta)
            else:
                positions[record.chunk_id] = len(self._ids)
                self._ids.append(record.chunk_id)
                self._vectors.append(list(record.vector))
                self._metas.append(dict(record.meta))
        self._sync_matrix()
        return len(records)

    def delete(self, chunk_ids: Sequence[int]) -> int:
        doomed = set(chunk_ids)
        keep = [i for i, cid in enumerate(self._ids) if cid not in doomed]
        removed = len(self._ids) - len(keep)
        if removed:
            self._assign([self._ids[i] for i in keep], [self._metas[i] for i in keep], [self._vectors[i] for i in keep])
        return removed

    def rebuild(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            self._check_dim(record.vector, f"chunk {record.chunk_id} ")
        self._assign([r.chunk_id for r in records], [dict(r.meta) for r in records], [list(r.vector) for r in records])

    def reset(self) -> None:
        self._assign([], [], [])

    def search(self, vector, allowed_chunk_ids, limit: int) -> list[IndexHit]:
        import numpy as np

        if self._matrix is None or not self._ids or limit <= 0:
            return []
        self._check_dim(vector, "查询")
        query = np.asarray(vector, dtype="float32").reshape(1, -1)
        scores = (self._matrix @ query.T).ravel()
        if allowed_chunk_ids is not None:
            wanted = set(int(cid) for cid in allowed_chunk_ids)
            if not wanted:
                return []
            allowed = np.fromiter((cid in wanted for cid in self._ids), dtype=bool, count=len(self._ids))
            if not allowed.any():
                return []
            scores = np.where(allowed, scores, -np.inf)
        top = np.argsort(-scores)[:limit]
        return [IndexHit(int(self._ids[i]), float(scores[i]), dict(self._metas[i])) for i in top if scores[i] > -np.inf]

    def stats(self) -> dict[str, Any]:
        return {"backend": self.backend, "dim": self.dim, "vectors": len(self._ids)}


class FaissIndex(NumpyIndex):
    """FAISS 精确内积检索（IndexFlatIP）。小规模知识库用近似索引是自找召回损失。

    FAISS 索引与自身 id 列表始终同步；需要按权限候选集过滤时，先取全量候选再过滤，
    避免"先取 top-k 再过滤"把合法结果一起丢掉。
    """

    backend = "faiss"

    def __init__(self, dim: int):
        super().__init__(dim)
        import faiss

        self._faiss = faiss
        self._index = None

    def _sync_matrix(self) -> None:

        super()._sync_matrix()
        if self._matrix is None:
            self._index = None
            return
        index = self._faiss.IndexFlatIP(self.dim)
        index.add(self._matrix)
        self._index = index

    def search(self, vector, allowed_chunk_ids, limit: int) -> list[IndexHit]:
        if self._index is None or not self._ids or limit <= 0:
            return []
        self._check_dim(vector, "查询")
        import numpy as np

        query = np.asarray(vector, dtype="float32").reshape(1, -1)
        wanted = None if allowed_chunk_ids is None else set(int(cid) for cid in allowed_chunk_ids)
        if wanted is not None and not wanted:
            return []
        k = self._index.ntotal if wanted is not None else min(limit, self._index.ntotal)
        distances, indices = self._index.search(query, max(1, k))
        hits: list[IndexHit] = []
        for score, i in zip(distances[0], indices[0], strict=True):
            if i < 0:
                continue
            chunk_id = int(self._ids[i])
            if wanted is not None and chunk_id not in wanted:
                continue
            hits.append(IndexHit(chunk_id, float(score), dict(self._metas[i])))
            if len(hits) >= limit:
                break
        return hits

    def stats(self) -> dict[str, Any]:
        return {**super().stats(), "faiss_total": int(getattr(self._index, "ntotal", 0))}


class PgVectorIndex:
    """把向量托管给 PostgreSQL 的 pgvector 扩展，排序在数据库里完成。

    `<=>` 是 pgvector 的余弦距离（越小越相似），这里转成 `1 - 距离` 保持
    "分数越大越相关"的统一语义。向量以字面量拼接，数值全部由 `float()` 转换，
    控件字符无法进入 SQL。
    """

    backend = "pgvector"

    def __init__(self, session, dim: int, provider: str, table: str = "chunk_embeddings"):
        self._session = session
        self.dim = dim
        self.provider = provider
        self._table = table

    @staticmethod
    def is_available(session) -> bool:
        from sqlalchemy import text

        try:
            if session.get_bind().dialect.name != "postgresql":
                return False
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            session.commit()
            return True
        except Exception as exc:  # noqa: BLE001 - 扩展缺失或权限不足都视为不可用
            session.rollback()
            logger.warning("pgvector 不可用：%s", str(exc).splitlines()[0][:160])
            return False

    def _literal(self, vector: Sequence[float]) -> str:
        if len(vector) != self.dim:
            raise ValueError(f"向量维度 {len(vector)} 与索引维度 {self.dim} 不一致")
        return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        from sqlalchemy import text

        if not records:
            return 0
        payload = [
            {
                "chunk_id": r.chunk_id,
                "embedding": self._literal(r.vector),
                "meta": json.dumps(r.meta, ensure_ascii=False),
                "provider": self.provider,
                "dim": self.dim,
            }
            for r in records
        ]
        self._session.execute(
            text(
                f"INSERT INTO {self._table} (chunk_id, provider, dim, content_hash, embedding, meta) "
                "VALUES (:chunk_id, :provider, :dim, '', CAST(:embedding AS vector), CAST(:meta AS jsonb)) "
                "ON CONFLICT (chunk_id) DO UPDATE SET provider = EXCLUDED.provider, dim = EXCLUDED.dim, "
                "embedding = EXCLUDED.embedding, meta = EXCLUDED.meta"
            ),
            payload,
        )
        self._session.commit()
        return len(records)

    def delete(self, chunk_ids: Sequence[int]) -> int:
        from sqlalchemy import text

        if not chunk_ids:
            return 0
        result = self._session.execute(
            text(f"DELETE FROM {self._table} WHERE chunk_id = ANY(:ids) AND provider = :provider AND dim = :dim"),
            {"ids": [int(c) for c in chunk_ids], "provider": self.provider, "dim": self.dim},
        )
        self._session.commit()
        return int(result.rowcount or 0)

    def rebuild(self, records: Sequence[VectorRecord]) -> None:
        from sqlalchemy import text

        self._session.execute(
            text(f"DELETE FROM {self._table} WHERE provider = :provider AND dim = :dim"),
            {"provider": self.provider, "dim": self.dim},
        )
        self._session.commit()
        self.upsert(records)

    def reset(self) -> None:
        self.rebuild([])

    def search(self, vector, allowed_chunk_ids, limit: int) -> list[IndexHit]:
        from sqlalchemy import text

        if limit <= 0:
            return []
        literal = self._literal(vector)
        params: dict[str, Any] = {"q": literal, "limit": limit, "provider": self.provider, "dim": self.dim}
        where = "WHERE provider = :provider AND dim = :dim"
        if allowed_chunk_ids is not None:
            ids = [int(c) for c in allowed_chunk_ids]
            if not ids:
                return []
            where += " AND chunk_id = ANY(:ids)"
            params["ids"] = ids
        rows = self._session.execute(
            text(
                f"SELECT chunk_id, 1 - ((embedding::vector({self.dim})) <=> CAST(:q AS vector({self.dim}))) AS score, meta "
                f"FROM {self._table} {where} "
                f"ORDER BY (embedding::vector({self.dim})) <=> CAST(:q AS vector({self.dim})) LIMIT :limit"
            ),
            params,
        ).all()
        hits: list[IndexHit] = []
        for chunk_id, score, meta in rows:
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            hits.append(IndexHit(int(chunk_id), float(score), meta or {}))
        return hits

    def stats(self) -> dict[str, Any]:
        from sqlalchemy import text

        try:
            count = self._session.execute(
                text(f"SELECT count(*) FROM {self._table} WHERE provider = :provider AND dim = :dim"),
                {"provider": self.provider, "dim": self.dim},
            ).scalar()
        except Exception:  # noqa: BLE001
            self._session.rollback()
            count = None
        return {"backend": self.backend, "dim": self.dim, "vectors": count}


def build_index(session, dim: int, provider: str = "unknown") -> VectorIndex:
    """选型顺序：pgvector → faiss → numpy。返回的索引初始为空，由调用方填充。"""
    if PgVectorIndex.is_available(session):
        return PgVectorIndex(session, dim, provider)
    try:
        import faiss  # noqa: F401
    except ImportError:
        logger.info("未安装 faiss-cpu，向量检索使用 NumPy 暴力检索")
        return NumpyIndex(dim)
    return FaissIndex(dim)
