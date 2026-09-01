"""Agent loop: authorization-safe retrieval tool + optional Qwen Responses API."""
import json
import re
from dataclasses import dataclass

from openai import OpenAI, OpenAIError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Chunk, Document, User

# 密级只能向下访问：CONFIDENTIAL 可以读三类，INTERNAL 不能读 CONFIDENTIAL。
RANK = {"PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2}

# 这是“审批前置”规则。它不执行操作，只创建审批单。
RISKY_PATTERNS = {
    "EXPORT": ("导出", "export", "下载全部"),
    "DELETE": ("删除", "delete", "清除"),
    "SEND": ("发送", "发给", "分享", "email"),
}


@dataclass
class SearchHit:
    document_id: int
    title: str
    excerpt: str
    score: int


def risk_action(message: str) -> str | None:
    """发现高风险词返回操作类型；普通提问返回 None。"""
    normalized = message.lower()
    for action, words in RISKY_PATTERNS.items():
        if any(word in normalized for word in words):
            return action
    return None


def tokenize(text: str) -> set[str]:
    """中文按字、英文按词做原型检索；生产环境应替换为 embedding + reranker。"""
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return set(chinese_chars + latin_words)


def search_knowledge(db: Session, user: User, query: str, limit: int = 4) -> list[SearchHit]:
    """先执行授权判断，再做匹配；模型永远拿不到越权 chunk。"""
    query_terms = tokenize(query)
    rows = db.execute(select(Chunk, Document).join(Document, Chunk.document_id == Document.id)).all()
    hits: list[SearchHit] = []
    for chunk, document in rows:
        # 第一道边界：密级必须不高于用户 clearance。
        if RANK[document.sensitivity] > RANK[user.clearance]:
            continue
        # 第二道边界：文档属于所有部门，或属于用户自己的部门。
        if document.department not in ("all", user.department):
            continue
        score = len(query_terms & tokenize(chunk.content + " " + document.title))
        # 中文单字会偶然重合；至少 3 个共同项才算命中。
        if score >= 3:
            hits.append(SearchHit(document.id, document.title, chunk.content[:400], score))
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]


TOOL = {
    "type": "function",
    "name": "search_knowledge",
    "description": "Search only documents the current employee is allowed to read. Return cited snippets before answering policy question.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Chinese or English search query"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}


def citations_from_hits(hits: list[SearchHit]) -> list[dict]:
    """将数据库命中转换成 API 返回和工具返回使用的来源格式。"""
    return [{"document_id": h.document_id, "title": h.title, "excerpt": h.excerpt} for h in hits]


def local_answer(hits: list[SearchHit]) -> str:
    """没有 API Key 时的可测试降级回答。"""
    if not hits:
        return "未在你有权限访问的知识库中找到依据。请联系文档负责人，或由管理员确认是否需要授权。"
    evidence = "\n".join(f"-{hit.title}: {hit.excerpt}" for hit in hits[:3])
    return f"以下是根据已授权知识库检索到的内容（请以来源原文为准）：\n{evidence}"


def qwen_request_options(reasoning_effort: str, *, initial_request: bool) -> dict:
    options: dict = {"reasoning": {"effort": reasoning_effort}}
    if initial_request:
        # 百炼思考模式不接受 required/object；关闭思考时才强制调用唯一工具。
        options["tool_choice"] = "required" if reasoning_effort == "none" else "auto"
    return options


def test_qwen_connection(api_key: str) -> tuple[bool, str]:
    """Make a minimal upstream request without storing or logging the supplied key."""
    settings = get_settings()
    try:
        client = OpenAI(api_key=api_key, base_url=settings.dashscope_base_url, timeout=15.0)
        client.responses.create(
            model=settings.qwen_model,
            input=[{"role": "user", "content": "只回复 OK"}],
            **qwen_request_options("none", initial_request=False),
        )
    except OpenAIError as exc:
        detail = str(exc).splitlines()[0][:240] or "上游返回了错误"
        return False, f"连接失败：{detail}"
    except Exception:
        return False, "连接失败：无法访问百炼服务，请检查网络和 Base URL。"
    return True, "连接成功，API Key 可用。"


def run_agent(db: Session, user: User, message: str, api_key: str | None = None) -> tuple[str, list[dict], str]:
    """返回 answer、citations、mode；工具始终在服务端再次执行授权过滤。"""
    settings = get_settings()
    initial_hits = search_knowledge(db, user, message)
    configured_key = api_key or settings.dashscope_api_key
    if not configured_key:
        return local_answer(initial_hits), citations_from_hits(initial_hits), "local"

    # 百炼兼容 OpenAI SDK；API Key 与 base_url 必须来自同一百炼地域/业务空间。
    client = OpenAI(api_key=configured_key, base_url=settings.dashscope_base_url)
    instructions = (
        "You are AegisKB, an enterprise policy assistant. Answer in Chinese. "
        "You must call search_knowledge for policy facts, rely only on its results, cite document titles, "
        "and state when evidence is missing. Never claim you can bypass permissions or execute external actions."
    )
    conversation: list[dict] = [{"role": "user", "content": message}]
    try:
        response = client.responses.create(
            model=settings.qwen_model,
            instructions=instructions,
            input=conversation,
            tools=[TOOL],
            **qwen_request_options(settings.qwen_reasoning_effort, initial_request=True),
        )
    except OpenAIError:
        return local_answer(initial_hits), citations_from_hits(initial_hits), "local"
    all_hits = initial_hits
    tool_called = False
    # 限制最多 3 轮工具调用，防止异常循环持续消耗 token。
    for _ in range(3):
        calls = [item for item in response.output if item.type == "function_call" and item.name == "search_knowledge"]
        if not calls:
            if not tool_called:
                return local_answer(initial_hits), citations_from_hits(initial_hits), "local"
            return response.output_text, citations_from_hits(all_hits), "qwen"
        tool_called = True
        for item in response.output:
            if item.type == "reasoning":
                conversation.append(item.model_dump(exclude_none=True))
                continue
            if item.type != "function_call" or item.name != "search_knowledge":
                continue
            call = item
            args = json.loads(call.arguments)
            all_hits = search_knowledge(db, user, args["query"])
            payload = {"results": citations_from_hits(all_hits), "access_enforced": True}
            # 百炼要求 function_call_output 紧随对应的 function_call 传回。
            conversation.append({"type": "function_call", "name": call.name, "arguments": call.arguments, "call_id": call.call_id})
            conversation.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(payload, ensure_ascii=False)})
        try:
            response = client.responses.create(
                model=settings.qwen_model,
                input=conversation,
                instructions=instructions,
                tools=[TOOL],
                **qwen_request_options(settings.qwen_reasoning_effort, initial_request=False),
            )
        except OpenAIError:
            return local_answer(all_hits), citations_from_hits(all_hits), "local"
    return "工具调用达到安全上限，已停止本次请求。请缩小问题范围后重试。", citations_from_hits(all_hits), "qwen"
