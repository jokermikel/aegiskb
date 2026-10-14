from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 环境变量名会自动按字段名的大写形式读取，例如 database_url 对应 DATABASE_URL。
    database_url: str = "sqlite:///./aegiskb.db"
    jwt_secret: str = "local-development-secret-change-me"
    # 百炼 OpenAI 兼容接口的认证和路由配置。
    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.8-max"
    qwen_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] = "none"
    # 单次模型请求的超时与重试，避免上游卡住时请求无限期挂起。
    qwen_timeout_seconds: float = Field(default=60.0, gt=0)
    qwen_max_retries: int = Field(default=1, ge=0)
    # 向量检索：auto 表示有 Key 时用 DashScope embedding，否则用确定性的本地向量，
    # 保证全新检出、无网络时依然可用。
    embedding_provider: Literal["auto", "dashscope", "offline"] = "auto"
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = Field(default=1024, gt=0)
    embedding_batch_size: int = Field(default=10, gt=0)
    # 留空表示按 embedding 提供方采用默认阈值；语义向量与哈希向量的合适阈值并不相同。
    retrieval_min_score: float | None = None
    # 混合检索：先取语义向量与关键词两路候选，再用 RRF 融合加词面信号做轻量重排。
    retrieval_mode: Literal["hybrid", "vector"] = "hybrid"
    retrieval_candidate_multiplier: int = Field(default=5, ge=1)
    retrieval_rrf_k: int = Field(default=60, ge=1)
    retrieval_vector_weight: float = Field(default=1.0, ge=0)
    retrieval_keyword_weight: float = Field(default=1.0, ge=0)
    retrieval_rerank_keyword_boost: float = Field(default=0.01, ge=0)
    app_env: str = "development"
    # 是否允许启动时自动建表。生产环境应设为 false，改用 Alembic 迁移管理结构。
    auto_create_schema: bool = True

    # 告诉 Pydantic 从项目根目录的 .env 读取变量；多余变量忽略。
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("retrieval_min_score", mode="before")
    @classmethod
    def blank_score_is_unset(cls, value):
        """`.env` 里写成 `RETRIEVAL_MIN_SCORE=` 时视为未设置，而不是类型错误。"""
        return None if value == "" else value

    @model_validator(mode="after")
    def forbid_auto_schema_in_production(self) -> "Settings":
        """生产环境禁止靠 create_all 建表。

        自动建表会绕过 Alembic：库里读不到 alembic_version，迁移历史与真实结构
        从此脱钩，后续任何一次 `alembic upgrade` 都可能建出错误的结构。
        这类配置错误必须在启动阶段就炸掉，而不是等到某次上线才发现。
        """
        if self.app_env.strip().lower() in {"production", "prod"} and self.auto_create_schema:
            raise ValueError(
                "APP_ENV=production 时 AUTO_CREATE_SCHEMA 必须为 false；"
                "生产环境请改用 `alembic upgrade head` 管理数据库结构。"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    # 整个进程只创建一次配置对象，避免每个请求重复读 .env
    return Settings()
