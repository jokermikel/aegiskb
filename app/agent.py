"""Agent loop: authorization-safe retrieval tool + optional Qwen Responses API."""
import json
import re
from dataclasses import dataclass

from openai import OpenAI, OpenAIError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Chunk, Document, User

# 密级只能向下访问：CONFIDENTIAL 可以读三类，INTERNAL 不能读 CONFIDENTIAL。
RANK = {"PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2}

# 按 clearance 反查可访问的密级集合，供 SQL 侧把密级过滤直接下推。
CLEARANCE_SENSITIVITIES = {
    level: tuple(name for name, value in RANK.items() if value <= rank)
    for level, rank in RANK.items()
}

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


@dataclass
class SearchHit:
    document_id: int
    title: str
    excerpt: str
    score: int


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


def parse_departments(value: str) -> tuple[str, ...]:
    """把部门字段解析成选择器元组，支持多部门（逗号分隔）。"""
    return tuple(item.strip().lower() for item in (value or "").split(",") if item.strip())


def department_visible(document_department: str, user_department: str) -> bool:
    """文档部门可见性判定（权威口径）。

    支持三种写法：
    - ``all``            全员可见
    - ``engineering``    单部门
    - ``engineering,it`` 多部门；``engineering/*`` 表示该部门及其子树
    """
    user_department = (user_department or "").strip().lower()
    for selector in parse_departments(document_department):
        if selector == "all":
            return True
        if selector.endswith("/*"):
            parent = selector[:-2]
            if user_department == parent or user_department.startswith(f"{parent}/"):
                return True
        elif selector == user_department:
            return True
    return False


def tokenize(text: str) -> set[str]:
    """中文按字、英文按词做原型检索；生产环境应替换为 embedding + reranker。"""
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    latin_words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return set(chinese_chars + latin_words)


def search_knowledge(db: Session, user: User, query: str, limit: int = 4) -> list[SearchHit]:
    """先执行授权判断，再做匹配；模型永远拿不到越权 chunk。

    密级用 IN 精确下推；部门因为要支持多部门与子树通配，SQL 侧只做「超集」预筛，
    随后仍逐条调用 department_visible() 做权威校验，越权判定不会因为下推而放松。
    """
    query_terms = tokenize(query)
    department_tokens = {(user.department or "").strip().lower(), "all"}
    if "/" in (user.department or ""):
        # 子部门账号（如 engineering/backend）要能看到父部门文档。
        department_tokens.add(user.department.split("/", 1)[0].strip().lower())
    statement = (
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Document.sensitivity.in_(CLEARANCE_SENSITIVITIES[user.clearance]),
            or_(*[Document.department.like(f"%{token}%") for token in sorted(department_tokens) if token]),
        )
    )
    hits: list[SearchHit] = []
    for chunk, document in db.execute(statement).all():
        # 权威校验：SQL 预筛只是性能优化，越权判定以这里为准。
        if RANK[document.sensitivity] > RANK[user.clearance]:
            continue
        if not department_visible(document.department, user.department):
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
