from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./aegiskb.db"
    jwt_secret: str = "local-development-secret-change-me"
    # 百炼 OpenAI 兼容接口的认证和路由配置。
    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.8-max"
    qwen_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] = "none"
    # Vector retrieval. `auto` uses DashScope when a key is available and a
    # deterministic local embedder otherwise, so a fresh checkout is usable
    # without network access.
    embedding_provider: Literal["auto", "dashscope", "offline"] = "auto"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=10, gt=0)
    # Leave unset to use a provider-specific default. A fixed value is useful
    # when tuning a production corpus, but one threshold is not appropriate
    # for both semantic and hashing embeddings.
    retrieval_min_score: float | None = None
    # Hybrid retrieval first obtains semantic and lexical candidate lists, then
    # uses reciprocal-rank fusion plus a lexical boost to rerank their union.
    retrieval_mode: Literal["hybrid", "vector"] = "hybrid"
    retrieval_candidate_multiplier: int = Field(default=5, ge=1)
    retrieval_rrf_k: int = Field(default=60, ge=1)
    retrieval_vector_weight: float = Field(default=1.0, ge=0)
    retrieval_keyword_weight: float = Field(default=1.0, ge=0)
    retrieval_rerank_keyword_boost: float = Field(default=0.01, ge=0)
    app_env: str = "development"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("retrieval_min_score", mode="before")
    @classmethod
    def blank_score_is_unset(cls, value):
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
