"""检索服务：embedding 向量化 + 向量索引 + RBAC 前置过滤。

链路（每一步都在数据库侧完成权限裁剪，模型永远看不到越权文本）：

    query ──embed──► query_vector
                        │
    user ──clearance/department──► 候选 chunk_id（SQL 过滤，先于一切相似度计算）
                        │
                        └──► vector_index.search(query_vector, allowed_ids, limit)
                                  │
                                  └──► SearchHit(document_id, title, excerpt, score)

与旧词法检索的差别：
- 旧实现把全部 chunk 读进内存做单字交集计数，分数是"命中字数"，与语义无关；
- 现在分数是余弦相似度（1 - 余弦距离），中文近义表达、"报销额度"这类换种说法
  的问法都能召回，不再依赖字面重合。
- 权限仍然在**检索之前**由 SQL 裁剪候选集，向量库只对合法候选排序。
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from sqlalchemy import delete, func, select, text
from sqlalchemy.orm import Session

from .chunking import chunk_text
from .config import get_settings
from .db import Chunk, ChunkEmbedding, Document, User
from .embeddings import get_embedder
from .vectorstore import IndexHit, VectorIndex, VectorRecord, build_index

logger = logging.getLogger(__name__)

RANK = {"PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2}
EXCERPT_CHARS = 400

# 离线哈希向量与在线语义向量的分数分布不同，统一阈值必然踩空，因此按 provider 给默认值。
DEFAULT_MIN_SCORE = {"offline-hashing": 0.05}
DEFAULT_MIN_SCORE_FALLBACK = 0.20
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_WORD = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class SearchHit:
    """一次检索命中。`score` 是混合召回后的重排序分数，越大越相关。"""

    document_id: int
    title: str
    excerpt: str
    score: float


@dataclass
class RetrievalMetrics:
    """向量检索的可观测数据，写审计日志时带上，便于回答"为什么这次没召回"。"""

    backend: str = "unknown"
    embedder: str = "unknown"
    dim: int = 0
    candidates: int = 0
    scored: int = 0
    kept: int = 0
    top_score: float | None = None
    min_score: float = 0.0
    embedded_now: int = 0
    latency_ms: int = 0
    degraded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval_backend": self.backend,
            "embedder": self.embedder,
            "dim": self.dim,
            "candidates": self.candidates,
            "scored": self.scored,
            "kept": self.kept,
            "top_score": None if self.top_score is None else round(self.top_score, 4),
            "min_score": self.min_score,
            "embedded_now": self.embedded_now,
            "latency_ms": self.latency_ms,
            "degraded": self.degraded,
            **self.extra,
        }


def keyword_terms(text: str) -> set[str]:
    """CJK 二字组 + 英文词，用于可解释的关键词候选召回。"""
    normalized = text.lower()
    terms: set[str] = set()
    for run in _CJK_RUN.findall(normalized):
        if len(run) == 1:
            terms.add(run)
        else:
            terms.update(run[index : index + 2] for index in range(len(run) - 1))
    terms.update(_LATIN_WORD.findall(normalized))
    return terms


def keyword_overlap_score(query_terms: set[str], title: str, content: str) -> float:
    """标题匹配权重更高，作为融合后轻量重排的精确词信号。"""
    if not query_terms:
        return 0.0
    title_matches = len(query_terms.intersection(keyword_terms(title)))
    content_matches = len(query_terms.intersection(keyword_terms(content)))
    return float(title_matches * 3 + content_matches)


# --------------------------------------------------------------------------- #
# 权限裁剪：必须发生在 DB 查询阶段
# --------------------------------------------------------------------------- #
def visible_sensitivities(user: User) -> list[str]:
    level = RANK.get(user.clearance, RANK["INTERNAL"])
    return [name for name, rank in RANK.items() if rank <= level]


def visible_chunk_ids(db: Session, user: User) -> list[int]:
    """返回该用户有权读取的 chunk id。这是唯一的权限入口，检索与索引都复用它。"""
    document_ids = select(Document.id).where(
        Document.sensitivity.in_(visible_sensitivities(user)),
        Document.department.in_(("all", user.department)),
    )
    return list(db.scalars(select(Chunk.id).where(Chunk.document_id.in_(document_ids))).all())


# --------------------------------------------------------------------------- #
# 向量列编解码：SQLite 走 float32 二进制；PostgreSQL 使用 pgvector 可读的字面量。
# --------------------------------------------------------------------------- #
def uses_vector_column() -> bool:
    return not get_settings().database_url.startswith("sqlite")


def pack_vector(vector: list[float]) -> Any:
    if uses_vector_column():
        return "[" + ",".join(f"{float(v):.8f}" for v in vector) + "]"
    import numpy as np

    return np.asarray(vector, dtype="float32").tobytes()


def unpack_vector(stored: Any) -> list[float]:
    if stored is None:
        return []
    if isinstance(stored, (bytes, bytearray, memoryview)):
        import numpy as np

        return np.frombuffer(bytes(stored), dtype="float32").astype("float32").tolist()
    if isinstance(stored, str):  # 极端情况下驱动把 vector 当字符串返回
        return [float(part) for part in stored.strip("[]").split(",") if part]
    return [float(v) for v in stored]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def embedding_input(title: str, content: str) -> str:
    """Keep the text used for embedding and cache invalidation identical."""
    return f"{title}\n{content}"


# --------------------------------------------------------------------------- #
# 索引生命周期
# --------------------------------------------------------------------------- #
_INDEX_CACHE: dict[str, VectorIndex] = {}


def index_cache_key(db_url: str, dim: int, provider: str) -> str:
    return f"{db_url}|{dim}|{provider}"


def get_index(db: Session, metrics: RetrievalMetrics | None = None) -> VectorIndex:
    """拿到当前数据库对应的向量索引（内存后端按来源进程内复用）。"""
    embedder = get_embedder()
    settings = get_settings()
    index = build_index(db, embedder.dim, embedder.name)
    if index.backend != "pgvector":
        key = index_cache_key(settings.database_url, embedder.dim, embedder.name)
        cached = _INDEX_CACHE.get(key)
        if cached is not None and cached.backend == index.backend:
            index = cached
        else:
            _INDEX_CACHE[key] = index
    else:
        index = build_index(db, embedder.dim, embedder.name)
    if metrics is not None:
        metrics.backend = index.backend
        metrics.embedder = embedder.name
        metrics.dim = embedder.dim
        metrics.degraded = embedder.degraded
    return index


def reset_index_cache() -> None:
    """测试与配置变更后清空内存索引。"""
    _INDEX_CACHE.clear()


def _drop_stale_rows(db: Session, live_chunk_ids: set[int], stored_ids: set[int]) -> int:
    stale = stored_ids - live_chunk_ids
    if not stale:
        return 0
    db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(stale)))
    db.commit()
    return len(stale)


def ensure_index(db: Session, *, metrics: RetrievalMetrics | None = None, batch_size: int = 32) -> VectorIndex:
    """让索引与 chunks 表保持一致：补齐缺失向量、丢弃已删 chunk、复用未变更向量。

    只有内容变化的 chunk 才会重新调用 embedding，避免每次问答把整个知识库重嵌入一遍。
    """
    settings = get_settings()
    embedder = get_embedder()
    index = get_index(db, metrics)

    rows = db.execute(
        select(Chunk.id, Chunk.document_id, Chunk.content, Document.title)
        .join(Document, Chunk.document_id == Document.id)
    ).all()
    live = {row.id: row for row in rows}
    provider_before_embedding = embedder.name
    stored = {
        rec.chunk_id: rec
        for rec in db.scalars(
            select(ChunkEmbedding).where(
                ChunkEmbedding.provider == provider_before_embedding, ChunkEmbedding.dim == embedder.dim
            )
        ).all()
    }
    _drop_stale_rows(db, set(live), set(db.scalars(select(ChunkEmbedding.chunk_id)).all()))

    # 1) 复用：向量、维度、provider、内容哈希都对得上，直接装载，不再调用 embedding。
    reusable: list[VectorRecord] = []
    for chunk_id, record in stored.items():
        chunk = live.get(chunk_id)
        if chunk is None or record.content_hash != content_hash(embedding_input(chunk.title, chunk.content)):
            continue
        vector = unpack_vector(record.embedding)
        if len(vector) != embedder.dim:
            continue
        reusable.append(VectorRecord(chunk_id, vector, {"document_id": chunk.document_id, "title": chunk.title}))

    # 2) 需要新建向量的 chunk（含 provider/维度切换导致的失效）。
    pending = [chunk for chunk_id, chunk in live.items() if chunk_id not in {r.chunk_id for r in reusable}]
    fresh: list[VectorRecord] = []
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed([embedding_input(chunk.title, chunk.content) for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("embedding 返回条数与输入不一致")
        for chunk, vector in zip(batch, vectors):
            fresh.append(VectorRecord(chunk.id, vector, {"document_id": chunk.document_id, "title": chunk.title}))

    # A remote provider may fail halfway through a multi-batch rebuild. Do not
    # mix its vectors with the fallback provider: rerun once with the stable
    # fallback so every score in the index has the same representation.
    if embedder.name != provider_before_embedding:
        return ensure_index(db, metrics=metrics, batch_size=batch_size)

    # 3) 落库，供下次复用（只有内容变化的 chunk 才会走到这里）。
    _persist(db, fresh, native_pgvector=index.backend == "pgvector")
    if metrics is not None:
        metrics.embedded_now = len(fresh)

    # 4) 同步索引。pgvector 的索引就是数据库表本身，第 3 步落库即生效；
    #    内存索引需要保证"条数与内容"都与 chunks 表一致，不一致就整体重建。
    if index.backend == "pgvector":
        return index

    if int(index.stats().get("vectors") or 0) != len(live):
        index.rebuild(reusable + fresh)
    elif fresh:
        index.upsert(fresh)
    return index


def _persist(db: Session, records: Iterable[VectorRecord], *, native_pgvector: bool = False) -> None:
    """把新算出来的向量写回 chunk_embeddings，供下次复用。"""
    embedder = get_embedder()
    items = list(records)
    if native_pgvector:
        # The ORM maps PostgreSQL vectors as text so SQLite can keep its BLOB
        # fallback. Cast explicitly here to preserve the physical `vector`
        # column and avoid driver-dependent list adaptation.
        payload = []
        for record in items:
            chunk = db.get(Chunk, record.chunk_id)
            if chunk is None:
                continue
            payload.append(
                {
                    "chunk_id": record.chunk_id,
                    "provider": embedder.name,
                    "dim": embedder.dim,
                    "content_hash": content_hash(embedding_input(str(record.meta.get("title") or ""), chunk.content)),
                    "embedding": pack_vector(record.vector),
                    "meta": record.meta,
                }
            )
        if payload:
            db.execute(
                text(
                    "INSERT INTO chunk_embeddings (chunk_id, provider, dim, content_hash, embedding, meta) "
                    "VALUES (:chunk_id, :provider, :dim, :content_hash, CAST(:embedding AS vector), CAST(:meta AS jsonb)) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET provider = EXCLUDED.provider, dim = EXCLUDED.dim, "
                    "content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding, meta = EXCLUDED.meta, "
                    "updated_at = CURRENT_TIMESTAMP"
                ),
                payload,
            )
            db.commit()
        return

    for record in items:
        row = db.get(ChunkEmbedding, record.chunk_id)
        chunk = db.get(Chunk, record.chunk_id)
        if chunk is None:
            continue
        if row is None:
            row = ChunkEmbedding(chunk_id=record.chunk_id)
            db.add(row)
        row.dim = embedder.dim
        row.provider = embedder.name
        row.content_hash = content_hash(embedding_input(str(record.meta.get("title") or ""), chunk.content))
        row.meta = {"document_id": record.meta.get("document_id"), "title": record.meta.get("title")}
        row.embedding = pack_vector(record.vector)
    if items:
        db.commit()


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
def default_min_score(provider: str) -> float:
    settings = get_settings()
    if settings.retrieval_min_score is not None:
        return settings.retrieval_min_score
    return DEFAULT_MIN_SCORE.get(provider, DEFAULT_MIN_SCORE_FALLBACK)


def search_knowledge(
    db: Session,
    user: User,
    query: str,
    limit: int = 4,
    *,
    min_score: float | None = None,
    metrics: RetrievalMetrics | None = None,
) -> list[SearchHit]:
    """混合检索有权限的 chunk，再用 RRF 和关键词信号重排序。

    保持与旧词法实现相同的函数签名（agent 层与工具 schema 无需改动），
    额外参数均为可选，默认行为即生产行为。
    """
    started = time.perf_counter()
    metrics = metrics if metrics is not None else RetrievalMetrics()
    embedder = get_embedder()
    index = ensure_index(db, metrics=metrics)
    threshold = default_min_score(embedder.name) if min_score is None else min_score
    metrics.min_score = threshold

    allowed = visible_chunk_ids(db, user)
    metrics.candidates = len(allowed)
    if not allowed:
        metrics.latency_ms = int((time.perf_counter() - started) * 1000)
        return []

    provider_before_query = embedder.name
    query_vector = embedder.embed([query])[0]
    if embedder.name != provider_before_query:
        # The index may still contain vectors from the provider that just
        # failed. Rebuild it before comparing with the fallback query vector.
        index = ensure_index(db, metrics=metrics)
        query_vector = embedder.embed([query])[0]
        threshold = default_min_score(embedder.name) if min_score is None else min_score
        metrics.min_score = threshold
    settings = get_settings()
    candidate_limit = max(limit, limit * max(1, settings.retrieval_candidate_multiplier))
    vector_hits = index.search(query_vector, allowed, candidate_limit)
    vector_rank = {
        hit.chunk_id: rank
        for rank, hit in enumerate(vector_hits, start=1)
        if hit.score >= threshold
    }
    vector_scores = {hit.chunk_id: float(hit.score) for hit in vector_hits if hit.chunk_id in vector_rank}

    # The rows are fetched only after SQL authorization has produced `allowed`.
    # This prevents keyword recall from inspecting a document that vectors were
    # not permitted to rank either.
    rows = db.execute(
        select(Chunk, Document).join(Document, Chunk.document_id == Document.id).where(Chunk.id.in_(allowed))
    ).all()
    row_by_chunk_id = {chunk.id: (chunk, document) for chunk, document in rows}

    keyword_rank: dict[int, int] = {}
    keyword_scores: dict[int, float] = {}
    if settings.retrieval_mode == "hybrid":
        terms = keyword_terms(query)
        scored_keywords = sorted(
            (
                (chunk.id, keyword_overlap_score(terms, document.title, chunk.content))
                for chunk, document in rows
            ),
            key=lambda item: (-item[1], item[0]),
        )
        scored_keywords = [(chunk_id, score) for chunk_id, score in scored_keywords if score > 0][:candidate_limit]
        keyword_rank = {chunk_id: rank for rank, (chunk_id, _) in enumerate(scored_keywords, start=1)}
        keyword_scores = dict(scored_keywords)

    # Reciprocal-rank fusion merges independent candidate lists. The final
    # lexical boost is intentionally small: exact policy terms can break ties
    # without replacing semantic retrieval with brittle keyword matching.
    candidate_ids = set(vector_rank).union(keyword_rank)
    max_keyword_score = max(keyword_scores.values(), default=1.0)
    fused: list[tuple[int, float]] = []
    for chunk_id in candidate_ids:
        score = 0.0
        if chunk_id in vector_rank:
            score += settings.retrieval_vector_weight / (settings.retrieval_rrf_k + vector_rank[chunk_id])
        if chunk_id in keyword_rank:
            score += settings.retrieval_keyword_weight / (settings.retrieval_rrf_k + keyword_rank[chunk_id])
            score += settings.retrieval_rerank_keyword_boost * (keyword_scores[chunk_id] / max_keyword_score)
        fused.append((chunk_id, score))
    fused.sort(key=lambda item: (-item[1], -vector_scores.get(item[0], -1.0), -keyword_scores.get(item[0], 0.0), item[0]))
    metrics.scored = len(fused)
    metrics.extra.update(
        {
            "retrieval_mode": settings.retrieval_mode,
            "vector_candidates": len(vector_rank),
            "keyword_candidates": len(keyword_rank),
            "fused_candidates": len(fused),
            "reranked": bool(keyword_rank),
        }
    )

    results: list[SearchHit] = []
    for chunk_id, score in fused[:limit]:
        row = row_by_chunk_id.get(chunk_id)
        if row is None:
            continue
        chunk, document = row
        results.append(SearchHit(document.id, document.title, chunk.content[:EXCERPT_CHARS], float(score)))

    metrics.kept = len(results)
    metrics.top_score = results[0].score if results else None
    metrics.latency_ms = int((time.perf_counter() - started) * 1000)
    return results


def reindex_document(db: Session, document_id: int) -> int:
    """文档新增或更新后强制重算其向量，返回重建的 chunk 数。"""
    chunk_ids = list(db.scalars(select(Chunk.id).where(Chunk.document_id == document_id)).all())
    if chunk_ids:
        db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.chunk_id.in_(chunk_ids)))
        db.commit()
        reset_index_cache()
    return len(chunk_ids)


def index_stats(db: Session) -> dict[str, Any]:
    """运维接口用：当前索引后端、向量条数、待嵌入 chunk 数。"""
    embedder = get_embedder()
    index = build_index(db, embedder.dim, embedder.name)
    stats = index.stats()
    stored = db.scalar(
        select(func.count()).select_from(ChunkEmbedding).where(
            ChunkEmbedding.provider == embedder.name, ChunkEmbedding.dim == embedder.dim
        )
    )
    stats.update(
        {
            "embedder": embedder.name,
            "degraded": embedder.degraded,
            "stored_vectors": int(stored or 0),
            "chunks": int(db.scalar(select(func.count()).select_from(Chunk)) or 0),
            "min_score": default_min_score(embedder.name),
            "retrieval_mode": get_settings().retrieval_mode,
            "candidate_multiplier": get_settings().retrieval_candidate_multiplier,
        }
    )
    return stats


__all__ = [
    "SearchHit",
    "RetrievalMetrics",
    "search_knowledge",
    "ensure_index",
    "reindex_document",
    "index_stats",
    "visible_chunk_ids",
    "visible_sensitivities",
    "chunk_text",
    "reset_index_cache",
    "RANK",
]
