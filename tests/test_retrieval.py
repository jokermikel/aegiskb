"""向量检索的安全性与缓存行为。

本模块的核心断言是：**授权裁剪必须发生在相似度排序之前** ——
机密文档的向量不能成为无权用户的候选，无论语义上多么接近。
"""
import os

# 必须与其他测试模块使用同一个库，且固定离线向量后端以保证结果确定。
os.environ["DATABASE_URL"] = "sqlite:///test_aegiskb.db"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["EMBEDDING_PROVIDER"] = "offline"

from sqlalchemy import delete, select

from app.db import ChunkEmbedding, SessionLocal, User
from app.embeddings import reset_embedder
from app.main import prepare_database, seed_demo_data
from app.retrieval import RetrievalMetrics, reset_index_cache, search_knowledge


def _user(db, email: str) -> User:
    user = db.scalar(select(User).where(User.email == email))
    assert user is not None, f"演示账号 {email} 不存在"
    return user


def setup_module():
    prepare_database()
    with SessionLocal() as db:
        seed_demo_data(db)


def test_vector_retrieval_enforces_rbac_before_ranking_and_caches_embeddings():
    """机密向量绝不能成为普通员工的候选；同时验证向量缓存命中。"""
    reset_embedder()
    reset_index_cache()
    with SessionLocal() as db:
        # 从空缓存开始，让「本次是否新嵌入」可判定。
        db.execute(delete(ChunkEmbedding))
        db.commit()

        finance = _user(db, "finance@example.com")
        employee = _user(db, "employee@example.com")

        first = RetrievalMetrics()
        finance_hits = search_knowledge(db, finance, "国内出差的住宿报销上限", metrics=first)
        assert first.backend in {"faiss", "numpy"}
        assert first.embedded_now > 0, "首次检索应产生新的向量"
        assert any(hit.title == "2026 差旅报销制度" for hit in finance_hits)
        assert all(isinstance(hit.score, float) for hit in finance_hits)
        assert first.extra["retrieval_mode"] == "hybrid"
        assert first.extra["vector_candidates"] > 0
        assert first.extra["keyword_candidates"] > 0
        assert first.extra["fused_candidates"] >= 1

        # 第二次检索应复用缓存，不再重复嵌入。
        second = RetrievalMetrics()
        search_knowledge(db, finance, "住宿报销标准", metrics=second)
        assert second.embedded_now == 0, "相同内容不应重复嵌入"

        # 同一批向量对无权用户必须整体不可见。
        employee_hits = search_knowledge(db, employee, "国内出差的住宿报销上限")
        assert all(hit.title != "2026 差旅报销制度" for hit in employee_hits)


def test_reindex_document_invalidates_cache_and_next_search_reembeds_once():
    """`reindex_document` 的职责是作废旧向量，重嵌入发生在下一次检索。

    这点容易被误解：它不是「立刻嵌入」，而是「标记失效 + 清空索引缓存」；
    随后第一次检索会由 ensure_index() 补上缺失的向量，且只补一次。
    """
    from app.db import Chunk, Document
    from app.retrieval import reindex_document

    db = SessionLocal()
    try:
        admin = _user(db, "admin@example.com")
        document = Document(
            title="向量重算验证文档",
            department="all",
            sensitivity="INTERNAL",
            body="第一版内容：本条描述的是初始状态下的管理要求，正文长度满足最小要求。",
            created_by=admin.id,
        )
        db.add(document)
        db.flush()
        db.add(Chunk(document_id=document.id, ordinal=0, content=document.body))
        db.commit()
        try:
            chunk_ids = [row.id for row in db.query(Chunk).filter_by(document_id=document.id).all()]
            assert len(chunk_ids) == 1

            def stored_for_document() -> int:
                return db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_id.in_(chunk_ids)).count()

            reset_embedder()
            reset_index_cache()
            search_knowledge(db, admin, "向量重算验证文档")
            assert stored_for_document() == 1, "首次检索应为其建立向量"

            # 作废后向量应被清空
            reindex_document(db, document.id)
            assert stored_for_document() == 0, "reindex 应清空该文档的向量"

            # 下一次检索补齐，且只补一次
            reset_index_cache()
            search_knowledge(db, admin, "向量重算验证文档")
            assert stored_for_document() == 1
            search_knowledge(db, admin, "向量重算验证文档")
            assert stored_for_document() == 1, "重复检索不应重复嵌入"
        finally:
            db.delete(document)
            db.commit()
    finally:
        db.close()
