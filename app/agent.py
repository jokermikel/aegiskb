"""Agent loop: authorization-safe retrieval tool + optional Qwen Responses API."""
import json

from openai import OpenAI, OpenAIError
from sqlalchemy.orm import Session

from .config import get_settings
from .db import User
from .retrieval import RetrievalMetrics, SearchHit, search_knowledge
RISKY_PATTERNS = {
    "EXPORT": ("导出", "export", "下载全部"),
    "DELETE": ("删除", "delete", "清除"),
    "SEND": ("发送", "发给", "分享", "email"),
}


def risk_action(message: str) -> str | None:
    normalized = message.lower()
    for action, words in RISKY_PATTERNS.items():
        if any(word in normalized for word in words):
            return action
    return None


TOOL = {
    "type": "function",
    "name": "search_knowledge",
    "description": "Search only documents the current employee is allowed to read. Return cited snippets before answering policy questions.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Chinese or English search query"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}


def citations_from_hits(hits: list[SearchHit]) -> list[dict]:
    return [{"document_id": h.document_id, "title": h.title, "excerpt": h.excerpt} for h in hits]


def local_answer(hits: list[SearchHit]) -> str:
    if not hits:
        return "未在你有权限访问的知识库中找到依据。请联系文档负责人，或由管理员确认是否需要授权。"
    evidence = "\n".join(f"- {hit.title}：{hit.excerpt}" for hit in hits[:3])
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


def run_agent(
    db: Session,
    user: User,
    message: str,
    api_key: str | None = None,
    *,
    retrieval_metrics: RetrievalMetrics | None = None,
) -> tuple[str, list[dict], str]:
    """Returns answer, citations and mode. The tool is server-side and always enforces RBAC."""
    settings = get_settings()
    metrics = retrieval_metrics if retrieval_metrics is not None else RetrievalMetrics()
    initial_hits = search_knowledge(db, user, message, metrics=metrics)
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
        # 密钥失效、额度耗尽或上游暂时不可用时，仍返回权限过滤后的本地答案。
        return local_answer(initial_hits), citations_from_hits(initial_hits), "local"
    all_hits = initial_hits
    tool_called = False
    # A bounded loop prevents runaway cost/tool calls.
    for _ in range(3):
        calls = [item for item in response.output if item.type == "function_call" and item.name == "search_knowledge"]
        if not calls:
            if not tool_called:
                # 思考模式只能使用 auto；模型跳过工具时退回服务端检索，避免无依据回答。
                return local_answer(initial_hits), citations_from_hits(initial_hits), "local"
            return response.output_text, citations_from_hits(all_hits), "qwen"
        tool_called = True
        for item in response.output:
            if item.type == "reasoning":
                # 思考模式的后续工具轮次需要带回上一轮 reasoning 项。
                conversation.append(item.model_dump(exclude_none=True))
                continue
            if item.type != "function_call" or item.name != "search_knowledge":
                continue
            call = item
            args = json.loads(call.arguments)
            all_hits = search_knowledge(db, user, args["query"], metrics=metrics)
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
