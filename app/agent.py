"""Agent loop: authorization-safe retrieval tool + optional Qwen Responses API.

检索本身由 app/retrieval.py 提供（混合召回 + 授权裁剪），本模块只负责
「高风险意图判定」与「模型工具调用循环」两件事。
"""
import json

from openai import OpenAI, OpenAIError
from sqlalchemy.orm import Session

from .config import get_settings
from .db import User
from .retrieval import (
    RANK,
    RetrievalMetrics,
    SearchHit,
    department_visible,
    parse_departments,
    search_knowledge,
)

__all__ = [
    "RANK",
    "RetrievalMetrics",
    "SearchHit",
    "citations_from_hits",
    "department_visible",
    "local_answer",
    "parse_departments",
    "qwen_request_options",
    "risk_action",
    "run_agent",
    "search_knowledge",
    "test_qwen_connection",
]

# 这是“审批前置”规则。它不执行操作，只创建审批单。
RISKY_PATTERNS = {
    "EXPORT": ("导出", "export", "下载全部"),
    "DELETE": ("删除", "delete", "清除"),
    "SEND": ("发送", "发给", "分享", "email"),
}

# 「询问制度」标记：命中说明用户在问规则，而不是让系统执行动作。
# 缺少这层判断时，「导出数据的审批流程是怎样的？」这类制度咨询会被误判成
# 高风险操作直接生成审批单，问题反而永远得不到回答。
CONSULTATION_MARKERS = (
    "流程", "步骤", "规定", "制度", "政策", "规范", "要求", "条件", "条款",
    "指引", "办法", "时限", "期限", "范围", "定义", "标准", "是什么",
    "什么是", "什么意思", "怎么", "怎样", "如何", "为什么", "为何",
    "哪些", "是否", "能否", "可以吗", "吗", "呢", "?", "？",
)

# 「直接下达指令」标记：判定优先级高于上面的疑问标记。
# 用于堵住「帮我把客户名单发出去好吗」这类用疑问语气绕过审批前置的写法。
DIRECTIVE_MARKERS = (
    "帮我", "请帮", "替我", "给我", "麻烦", "立刻", "马上", "现在就",
)


def is_consultation(message: str) -> bool:
    """判断消息是「询问制度」还是「下达指令」；指令标记优先。"""
    normalized = message.lower()
    if any(marker in normalized for marker in DIRECTIVE_MARKERS):
        return False
    return any(marker in normalized for marker in CONSULTATION_MARKERS)


def risk_action(message: str) -> str | None:
    """发现高风险指令时返回操作类型；制度咨询与普通提问返回 None。

    只有「命中高风险动作词」且「不是制度咨询」时才判定为高风险操作。
    """
    normalized = message.lower()
    action = next(
        (candidate for candidate, words in RISKY_PATTERNS.items() if any(word in normalized for word in words)),
        None,
    )
    if action is None or is_consultation(message):
        return None
    return action


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


def build_client(api_key: str) -> OpenAI:
    """统一构造客户端：必须带超时与重试上限，避免上游卡住时请求无限期挂起。"""
    settings = get_settings()
    return OpenAI(
        api_key=api_key,
        base_url=settings.dashscope_base_url,
        timeout=settings.qwen_timeout_seconds,
        max_retries=settings.qwen_max_retries,
    )


def test_qwen_connection(api_key: str) -> tuple[bool, str]:
    """Make a minimal upstream request without storing or logging the supplied key."""
    settings = get_settings()
    try:
        client = build_client(api_key)
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
    retrieval_metrics: RetrievalMetrics | None = None,
) -> tuple[str, list[dict], str]:
    """返回 answer、citations、mode；工具始终在服务端再次执行授权过滤。

    ``retrieval_metrics`` 由调用方传入时，检索层会把后端、耗时、候选数等写入其中，
    便于把「为什么这次没召回」一并记进审计日志。
    """
    settings = get_settings()
    initial_hits = search_knowledge(db, user, message, metrics=retrieval_metrics)
    configured_key = api_key or settings.dashscope_api_key
    if not configured_key:
        return local_answer(initial_hits), citations_from_hits(initial_hits), "local"

    # 百炼兼容 OpenAI SDK；API Key 与 base_url 必须来自同一百炼地域/业务空间。
    client = build_client(configured_key)
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
    except Exception:
        # 上游超时、限流或网络异常统一降级到本地检索结果，不让请求失败。
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
            all_hits = search_knowledge(db, user, args["query"], metrics=retrieval_metrics)
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
        except Exception:
            return local_answer(all_hits), citations_from_hits(all_hits), "local"
    return "工具调用达到安全上限，已停止本次请求。请缩小问题范围后重试。", citations_from_hits(all_hits), "qwen"
