"""文本清洗与分块。

分块策略从"固定窗口硬切"改为"按句边界打包"：
- 固定窗口会把一句话从中间切开，两半各自和原意都不像，向量也会被拉偏；
- 中文按 `。！？；` 切句，英文按 `. ! ? ;` 与换行切句，再把句子顺序打包进
  不超过 `size` 的窗口，相邻窗口保留 `overlap` 字符的尾部上下文。
- 单句超长（无标点的长日志、表格转文本）时硬切，保证不会产出超长 chunk。
"""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；;!?])|(?<=\.)\s+|\n+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part and part.strip()]
    return parts or [text.strip()]


def _hard_split(sentence: str, size: int) -> list[str]:
    return [sentence[i : i + size] for i in range(0, len(sentence), size)]


def _tail(text: str, overlap: int) -> str:
    """取尾部 `overlap` 字符作为下一窗口的上下文，尽量从句子边界开始。"""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    for boundary in ("。", "！", "？", "；", ".", "!", "?", ";"):
        idx = tail.find(boundary)
        if 0 <= idx < len(tail) - 1:
            return tail[idx + 1 :]
    return tail


def chunk_text(text: str, size: int = 350, overlap: int = 60) -> list[str]:
    """把整篇文本切成有重叠的 chunk。空白被压缩，避免向量里塞满无意义字符。"""
    if size <= 0:
        raise ValueError("size 必须为正整数")
    overlap = max(0, min(overlap, size - 1))
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= size:
        return [clean]

    sentences: list[str] = []
    for sentence in split_sentences(clean):
        sentences.extend(_hard_split(sentence, size) if len(sentence) > size else [sentence])

    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer}{sentence}" if not buffer else f"{buffer} {sentence}"
        if len(candidate) <= size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
            buffer = f"{_tail(buffer, overlap)}{sentence}".strip()
        else:
            buffer = sentence
        while len(buffer) > size:  # 句子本身超长时的兜底
            chunks.append(buffer[:size])
            buffer = f"{_tail(buffer[:size], overlap)}{buffer[size:]}".strip()
    if buffer:
        chunks.append(buffer)
    return [chunk for chunk in chunks if chunk]


def estimate_tokens(text: str) -> int:
    """粗估 token 数：CJK 约 1 字 1 token，其余按 4 字符 1 token。仅用于统计展示。"""
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return cjk + max(0, other) // 4
