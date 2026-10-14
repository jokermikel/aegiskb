"""Embedding 提供方：百炼 text-embedding 在线向量 + 离线确定性向量兜底。

设计取舍：
- 检索质量优先使用真正的语义向量（百炼 text-embedding-v3/v4）；
- 但单元测试、CI 与离线演示不应该依赖外部网络和有效密钥，因此提供
  `offline-hashing` 兜底实现。两者输出的都是 **同维度的 L2 归一化向量**，
  所以下游的存储层与索引层不需要知道当前用的是哪一种。
- 每条向量都会记录 `provider` 与 `dim`。当在线/离线切换导致向量空间变化时，
  查询侧能检测到不一致并触发重嵌入，避免拿两套空间的向量做内积这种静默错误。
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

from .config import get_settings

logger = logging.getLogger(__name__)

_HAN = re.compile(r"[\u4e00-\u9fff]")
_HAN_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")
_LATIN = re.compile(r"[a-z0-9_]+")


class EmbeddingError(RuntimeError):
    """embedding 提供方返回了不可用的结果（维度不符、空向量等）。"""


def l2_normalize(vec: list[float]) -> list[float]:
    """L2 归一化：归一化后内积 == 余弦相似度，pgvector/FAISS/NumPy 三条路径才能统一打分。"""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


class Embedder(Protocol):
    """所有 embedding 实现的统一契约。"""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashingEmbedder:
    """离线确定性向量：字符 n-gram + 带符号哈希（hashing trick）。

    它不是语义模型——"差旅" 和 "报销" 之间没有学习到的语义关联，靠的是中文
    二字组与单字的表面重合，因此召回质量明显低于真实 embedding。它的价值在于
    确定、无需网络、无需下载权重，让权限过滤和检索链路在没有密钥时依然可跑、可测。
    """

    name = "offline-hashing"

    def __init__(self, dim: int = 1024):
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim

    @staticmethod
    def _features(text: str) -> list[str]:
        lowered = text.lower()
        feats = _HAN.findall(lowered)                      # 单字：保证任意问句都有信号
        for run in _HAN_RUN.findall(lowered):              # 二字组：承载短语级信息
            feats.extend(run[i : i + 2] for i in range(len(run) - 1))
        feats.extend(_LATIN.findall(lowered))              # 英文/数字按词切
        return feats

    def _one(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for feat in self._features(text):
            counts[feat] = counts.get(feat, 0) + 1
        vec = [0.0] * self.dim
        for feat, count in counts.items():
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            # 低位选桶、高位定符号：独立维度上取，降低哈希冲突带来的系统性偏置。
            vec[h % self.dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(count))
        return l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]


class DashScopeEmbedder:
    """百炼（DashScope）OpenAI 兼容 `/embeddings` 接口。"""

    name = "dashscope"

    def __init__(self, api_key: str, base_url: str, model: str, dim: int, batch_size: int = 10, timeout: float = 30.0):
        from openai import OpenAI

        self.dim = dim
        self.model = model
        self.batch_size = max(1, batch_size)
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self.model, input=batch, dimensions=self.dim, encoding_format="float"
                )
            except TypeError:
                # 老版本 SDK 不接受 dimensions 参数。
                response = self._client.embeddings.create(model=self.model, input=batch, encoding_format="float")
            ordered = sorted(response.data, key=lambda item: getattr(item, "index", 0))
            for item in ordered:
                vector = list(item.embedding)
                if len(vector) != self.dim:
                    raise EmbeddingError(f"返回维度 {len(vector)} 与配置的 {self.dim} 不一致")
                out.append(l2_normalize(vector))
        return out


class EmbeddingService:
    """在线优先、失败即降级的门面。

    降级只在**第一次失败**时发生并记录一条 warning：密钥失效或网络不通时，
    后续请求不会再反复打上游，避免每次问答都白等一次超时。
    """

    def __init__(self, primary: DashScopeEmbedder | None, fallback: HashingEmbedder):
        self._primary = primary
        self._fallback = fallback
        self.dim = primary.dim if primary else fallback.dim
        self._degraded = primary is None
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        if self._primary is not None and not self._degraded:
            return self._primary.name
        return self._fallback.name

    @property
    def degraded(self) -> bool:
        return self._primary is not None and self._degraded

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._primary is not None and not self._degraded:
            try:
                return self._primary.embed(texts)
            except Exception as exc:  # noqa: BLE001 - 任何上游异常都应降级而不是中断问答
                self._degraded = True
                self.last_error = str(exc).splitlines()[0][:240] if str(exc) else exc.__class__.__name__
                logger.warning("在线 embedding 不可用，已降级为 %s：%s", self._fallback.name, self.last_error)
        return self._fallback.embed(texts)

    def probe(self) -> bool:
        """启动时探活一次，把"能不能用在线向量"变成一条明确的日志而不是首次请求的偶发延迟。"""
        if self._primary is None:
            return False
        try:
            self._primary.embed(["探活"])
            self._degraded = False
            return True
        except Exception as exc:  # noqa: BLE001
            self._degraded = True
            self.last_error = str(exc).splitlines()[0][:240] if str(exc) else exc.__class__.__name__
            logger.warning("在线 embedding 探活失败，本次进程使用 %s：%s", self._fallback.name, self.last_error)
            return False


_service: EmbeddingService | None = None


def build_embedder() -> EmbeddingService:
    """按配置组装 embedding 服务。provider=auto 时有密钥就试在线，没有就直接离线。"""
    settings = get_settings()
    fallback = HashingEmbedder(dim=settings.embedding_dim)
    provider = settings.embedding_provider
    if provider == "offline":
        return EmbeddingService(None, fallback)
    if not settings.dashscope_api_key:
        if provider == "dashscope":
            logger.warning("EMBEDDING_PROVIDER=dashscope 但未配置 DASHSCOPE_API_KEY，改用离线向量")
        return EmbeddingService(None, fallback)
    primary = DashScopeEmbedder(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )
    return EmbeddingService(primary, fallback)


def get_embedder() -> EmbeddingService:
    """进程级单例：embedding 客户端有连接池，按请求重建是浪费。"""
    global _service
    if _service is None:
        _service = build_embedder()
    return _service


def reset_embedder() -> None:
    """测试与配置热更新用。"""
    global _service
    _service = None
