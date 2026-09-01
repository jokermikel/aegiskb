from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # 环境变量名会自动按字段名的大写形式读取，例如 database_url 对应 DATABASE_URL。
    database_url: str = "sqlite:///./aegiskb.db"
    jwt_secret: str = "local-development-secret-change-me"
    # 百炼 OpenAI 兼容接口的认证和路由配置。
    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen3.8-max"
    qwen_reasoning_effort:Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] = "none"
    app_env: str = "development"

    # 告诉 Pydantic 从项目根目录的 .env 读取变量；多余变量忽略。
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    # 整个进程只创建一次配置对象，避免每个请求重复读 .env
    return Settings()