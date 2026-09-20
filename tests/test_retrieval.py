import os

# Keep this module fully offline even if the developer has a local .env file.
os.environ["DATABASE_URL"] = "sqlite:///./test_aegiskb.db"
os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["EMBEDDING_PROVIDER"] = "offline"

from sqlalchemy import delete, select

from app.db import ChunkEmbedding, SessionLocal, User, initialize_database
from app.embeddings import reset_embedder
from app.main import seed_demo_data
from app.retrieval import RetrievalMetrics, reset_index_cache, search_knowledge


def _user(db, email: str) -> User:
    user = db.scalar(select(User).where(User.email == email))
    assert user is not None
    return user


def setup_module():
    initialize_database()
    with SessionLocal() as db:
        seed_demo_data(db)


def test_vector_retrieval_enforces_rbac_before_ranking_and_caches_embeddings():
    """A confidential vector must never become a candidate for an employee."""
    reset_embedder()
    reset_index_cache()
    with SessionLocal() as db:
        # Rebuilding from an empty cache makes cache behavior deterministic.
        db.execute(delete(ChunkEmbedding))
        db.commit()

        finance = _user(db, "finance@example.com")
        employee = _user(db, "employee@example.com")

        first = RetrievalMetrics()
        finance_hits = search_knowledge(db, finance, "国内出差的住宿报销上限", metrics=first)
        assert first.backend in {"faiss", "numpy"}
        assert first.embedded_now > 0
        assert any(hit.title == "2026 差旅报销制度" for hit in finance_hits)
        assert all(isinstance(hit.score, float) for hit in finance_hits)
        assert first.extra["retrieval_mode"] == "hybrid"
        assert first.extra["vector_candidates"] > 0
        assert first.extra["keyword_candidates"] > 0
        assert first.extra["fused_candidates"] >= first.extra["keyword_candidates"]

        second = RetrievalMetrics()
        search_knowledge(db, finance, "住宿报销标准", metrics=second)
        assert second.embedded_now == 0

        employee_hits = search_knowledge(db, employee, "国内出差的住宿报销上限")
        assert all(hit.title != "2026 差旅报销制度" for hit in employee_hits)
