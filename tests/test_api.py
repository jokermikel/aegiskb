import os

# 测试必须使用独立数据库，绝不能污染演示数据库。
os.environ["DATABASE_URL"] = "sqlite:///test_aegiskb.db"
os.environ["JWT_SECRET"] = "test-secret"
# 显式覆盖 .env 中可能存在的真实百炼密钥，保证测试只走本地模式。
os.environ["DASHSCOPE_API_KEY"] = ""
# 固定使用离线向量后端：结果确定、不依赖网络、不消耗 token。
os.environ["EMBEDDING_PROVIDER"] = "offline"

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def setup_module():
    # 进入 TestClient 生命周期，触发 startup，自动建表和导入演示数据。
    client.__enter__()


def teardown_module():
    client.__exit__(None, None, None)


def login(email: str) -> str:
    response = client.post("/api/v1/auth/login", json={"email": email, "password": "demo-password"})
    assert response.status_code == 200
    return response.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_clearance_filters_confidential_document():
    employee = login("employee@example.com")
    finance = login("finance@example.com")
    question = {"message": "国内出差住宿标准是多少"}
    employee_response = client.post("/api/v1/agent/chat", json=question, headers=auth(employee)).json()
    finance_response = client.post("/api/v1/agent/chat", json=question, headers=auth(finance)).json()
    # employee 不能得到任意机密文档引用。
    assert not employee_response["citations"]
    # finance 得到正确文档来源。
    assert any(item["title"] == "2026 差旅报销制度" for item in finance_response["citations"])


def test_approval_visibility_and_admin_decisions():
    employee = login("employee@example.com")
    finance = login("finance@example.com")
    admin = login("admin@example.com")
    current_admin = client.get("/api/v1/auth/me", headers=auth(admin))
    assert current_admin.status_code == 200
    assert current_admin.json()["role"] == "admin"
    result = client.post("/api/v1/agent/chat", json={"message": "请导出全部客户数据"}, headers=auth(employee)).json()
    assert result["requires_approval"] is True
    approval_id = result["approval_id"]

    # 请求人能看到自己的审批单，其他普通用户看不到。
    employee_items = client.get("/api/v1/approvals", headers=auth(employee))
    finance_items = client.get("/api/v1/approvals", headers=auth(finance))
    assert employee_items.status_code == 200
    employee_approval = next(item for item in employee_items.json() if item["id"] == approval_id)
    assert employee_approval["requester_email"] == "employee@example.com"
    assert employee_approval["payload"]["message"] == "请导出全部客户数据"
    assert all(item["id"] != approval_id for item in finance_items.json())

    # 管理员能看到所有用户的待审批记录。
    admin_pending = client.get("/api/v1/approvals", headers=auth(admin))
    assert admin_pending.status_code == 200
    assert any(item["id"] == approval_id and item["status"] == "PENDING" for item in admin_pending.json())
    # 请求人不是管理员，不能自己批准。
    forbidden = client.post(f"/api/v1/approvals/{approval_id}/decision", json={"decision": "APPROVED", "comment": "try"}, headers=auth(employee))
    assert forbidden.status_code == 403
    # admin 可以批准一次。
    approved = client.post(f"/api/v1/approvals/{approval_id}/decision", json={"decision": "APPROVED", "comment": "security review complete"}, headers=auth(admin))
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"

    # 已处理审批不能重复审批。
    duplicate = client.post(f"/api/v1/approvals/{approval_id}/decision", json={"decision": "REJECTED", "comment": "duplicate decision"}, headers=auth(admin))
    assert duplicate.status_code == 409

    # 管理员也能拒绝另一位用户提交的审批单。
    second = client.post("/api/v1/agent/chat", json={"message": "请删除这份客户数据"}, headers=auth(finance)).json()
    rejected = client.post(f"/api/v1/approvals/{second['approval_id']}/decision", json={"decision": "REJECTED", "comment": "not allowed"}, headers=auth(admin))
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "REJECTED"


def test_qwen_tool_choice_is_compatible_with_reasoning_mode():
    from app.agent import qwen_request_options

    assert qwen_request_options("none", initial_request=True) == {
        "reasoning": {"effort": "none"},
        "tool_choice": "required",
    }
    assert qwen_request_options("high", initial_request=True) == {
        "reasoning": {"effort": "high"},
        "tool_choice": "auto",
    }
    assert qwen_request_options("high", initial_request=False) == {
        "reasoning": {"effort": "high"},
    }


def test_qwen_api_key_test_endpoint(monkeypatch):
    from app import main

    finance = login("finance@example.com")
    seen = {}

    def fake_test(api_key: str):
        seen["api_key"] = api_key
        return True, "连接成功，API Key 可用。"

    monkeypatch.setattr(main, "test_qwen_connection", fake_test)
    response = client.post(
        "/api/v1/settings/qwen/test",
        json={"api_key": "sk-valid-demo"},
        headers=auth(finance),
    )
    assert response.status_code == 200
    assert response.json() == {
        "connected": True,
        "message": "连接成功，API Key 可用。",
        "model": "qwen3.8-max",
    }
    assert seen["api_key"] == "sk-valid-demo"


def test_qwen_api_key_test_requires_auth_and_valid_key_length():
    missing_auth = client.post("/api/v1/settings/qwen/test", json={"api_key": "sk-valid-demo"})
    # FastAPI HTTPBearer 默认对缺少 Authorization 头返回 403。
    assert missing_auth.status_code == 403

    employee = login("employee@example.com")
    too_short = client.post(
        "/api/v1/settings/qwen/test",
        json={"api_key": "short"},
        headers=auth(employee),
    )
    assert too_short.status_code == 422


def test_chat_passes_session_api_key_to_agent(monkeypatch):
    from app import main

    captured = {}

    def fake_run_agent(db, user, message, api_key=None, retrieval_metrics=None):
        captured.update(message=message, api_key=api_key)
        return "测试回答", [], "local"

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    employee = login("employee@example.com")
    response = client.post(
        "/api/v1/agent/chat",
        json={"message": "测试透传 Key"},
        headers={**auth(employee), "X-DASHSCOPE-API-KEY": "sk-session-demo"},
    )
    assert response.status_code == 200
    assert captured == {"message": "测试透传 Key", "api_key": "sk-session-demo"}


def test_qwen_error_falls_back_to_local_answer(monkeypatch):
    from openai import OpenAIError

    from app import agent
    from app.db import SessionLocal, User

    class FailingResponses:
        def create(self, **kwargs):
            raise OpenAIError()

    class FailingClient:
        def __init__(self, **kwargs):
            self.responses = FailingResponses()

    monkeypatch.setattr(agent, "OpenAI", FailingClient)
    db = SessionLocal()
    try:
        employee = db.query(User).filter_by(email="employee@example.com").one()
        answer, citations, mode = agent.run_agent(
            db, employee, "每周可以远程办公几天？", api_key="sk-failing-demo"
        )
    finally:
        db.close()

    assert mode == "local"
    assert "两天" in answer
    assert any(item["title"] == "远程办公制度" for item in citations)

def test_seed_dataset_volume():
    """演示数据集固定为 10 个账号、50 篇文档、100 条审批单，标题不重复。"""
    from app.seed_data import DEMO_DOCUMENTS, DEMO_USERS, build_approval_samples

    assert len(DEMO_USERS) == 10
    assert len(DEMO_DOCUMENTS) == 50
    titles = [title for title, *_ in DEMO_DOCUMENTS]
    assert len(set(titles)) == 50
    # 密级必须落在 schemas 声明的三档之内，否则权限比较会抛 KeyError。
    assert {sensitivity for _, _, sensitivity, _ in DEMO_DOCUMENTS} == {"PUBLIC", "INTERNAL", "CONFIDENTIAL"}
    # 文档部门必须都能对上一个演示账号，否则这些文档对谁都不可见。
    assert {department for _, department, _, _ in DEMO_DOCUMENTS} - {"all"} == {user[2] for user in DEMO_USERS}

    samples = build_approval_samples([email for email, *_ in DEMO_USERS])
    assert len(samples) == 100
    assert {sample["status"] for sample in samples} == {"PENDING", "APPROVED", "REJECTED"}
    # 已裁决的记录必须有意见和裁决时间，待处理的必须都没有。
    for sample in samples:
        decided = sample["status"] != "PENDING"
        assert (sample["comment"] is not None) is decided
        assert (sample["decided_at"] is not None) is decided


def test_seed_dataset_is_loaded_into_database():
    """空库初始化后数据确实落库，且每篇文档都切出了至少一个可检索片段。"""
    from app.db import ApprovalRequest, Chunk, Document, SessionLocal

    db = SessionLocal()
    try:
        documents = db.query(Document).all()
        assert len(documents) == 50
        assert db.query(Chunk).count() >= 50
        assert all(db.query(Chunk).filter(Chunk.document_id == document.id).count() >= 1 for document in documents)
        # 其他用例也会通过问答新增审批单，因此这里只校验下限。
        assert db.query(ApprovalRequest).count() >= 100
    finally:
        db.close()


def test_employee_never_receives_unauthorized_citations():
    """属性测试：无论问什么，返回的每条引用都必须是该用户有权阅读的文档。"""
    from app.agent import RANK, search_knowledge
    from app.db import Document, SessionLocal, User

    questions = [
        "国内出差住宿标准是多少",
        "薪酬结构和调薪机制是怎样的",
        "合同评审和用印要经过什么流程",
        "数据泄露之后该怎么上报",
        "采购付款需要谁审批",
        "密钥和证书多久轮换一次",
    ]
    db = SessionLocal()
    try:
        employee = db.query(User).filter_by(email="employee@example.com").one()
        for question in questions:
            for hit in search_knowledge(db, employee, question):
                document = db.get(Document, hit.document_id)
                assert RANK[document.sensitivity] <= RANK[employee.clearance]
                assert document.department in ("all", employee.department)
    finally:
        db.close()


def test_department_isolation_across_seeded_users():
    """不同部门的账号不能检索到其他部门的文档。"""
    from app.agent import search_knowledge
    from app.db import Document, SessionLocal, User

    db = SessionLocal()
    try:
        for email in ("hr@example.com", "sales@example.com", "ops@example.com", "it@example.com", "legal@example.com"):
            user = db.query(User).filter_by(email=email).one()
            for hit in search_knowledge(db, user, "制度流程与审批要求"):
                document = db.get(Document, hit.document_id)
                assert document.department in ("all", user.department)
    finally:
        db.close()


def test_seeded_documents_are_visible_to_every_demo_account():
    """每个演示账号至少能检索到一篇文档，否则演示时会出现空知识库。"""
    from app.agent import search_knowledge
    from app.db import SessionLocal, User

    db = SessionLocal()
    try:
        for user in db.query(User).all():
            assert search_knowledge(db, user, "制度管理流程与审批要求"), f"{user.email} 看不到任何文档"
    finally:
        db.close()


def test_policy_consultation_is_not_treated_as_risky_action():
    """制度咨询不得被判成高风险操作。

    此前 risk_action 只做关键词子串匹配，导致「导出数据的审批流程是怎样的？」
    这类制度咨询被误建成审批单，问题反而永远得不到回答。
    """
    from app.agent import risk_action

    consultations = [
        "导出数据的审批流程是怎样的？",
        "数据分级规范里提到的发送是什么意思",
        "清除浏览器缓存可以吗",
        "pdf 文件下载全部需要审批吗",
        "删除数据的制度依据是什么",
    ]
    for message in consultations:
        assert risk_action(message) is None, f"制度咨询被误判为高风险操作：{message}"

    directives = [
        "导出全部客户联系人",
        "请导出全部客户数据",
        "把合同发送到我的私人邮箱",
        "删除这份过期的客户数据",
        "将审计报告 email 给外部顾问",
    ]
    for message in directives:
        assert risk_action(message) is not None, f"高风险指令被错误放行：{message}"


def test_directive_dressed_as_a_question_still_requires_approval():
    """用疑问语气下达指令，不能绕过审批前置。"""
    from app.agent import risk_action

    assert risk_action("帮我把客户名单发给合作方好吗") == "SEND"
    assert risk_action("麻烦导出全部客户联系人可以吗") == "EXPORT"


def test_policy_consultation_reaches_the_agent(monkeypatch):
    """端到端确认：制度咨询会真的走到 Agent，而不是变成一张审批单。"""
    from app import main

    captured = {}

    def fake_run_agent(db, user, message, api_key=None, retrieval_metrics=None):
        captured["message"] = message
        return "制度回答", [], "local"

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    employee = login("employee@example.com")
    response = client.post(
        "/api/v1/agent/chat",
        json={"message": "导出数据的审批流程是怎样的？"},
        headers=auth(employee),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["requires_approval"] is False
    assert body["answer"] == "制度回答"
    assert captured["message"] == "导出数据的审批流程是怎样的？"


def test_audit_log_limit_cannot_be_bypassed():
    """limit 越界必须被参数校验拒绝：SQLite 中 LIMIT -1 等于不限行数。"""
    admin = login("admin@example.com")
    assert client.get("/api/v1/audit-logs", params={"limit": 50}, headers=auth(admin)).status_code == 200
    assert client.get("/api/v1/audit-logs", params={"limit": 200}, headers=auth(admin)).status_code == 200
    assert client.get("/api/v1/audit-logs", params={"limit": -1}, headers=auth(admin)).status_code == 422
    assert client.get("/api/v1/audit-logs", params={"limit": 0}, headers=auth(admin)).status_code == 422
    assert client.get("/api/v1/audit-logs", params={"limit": 500}, headers=auth(admin)).status_code == 422


def test_approval_list_is_paginated():
    """审批列表分页：总数走响应头，页与页之间不重复。"""
    admin = login("admin@example.com")
    first = client.get("/api/v1/approvals", params={"page": 1, "page_size": 10}, headers=auth(admin))
    assert first.status_code == 200
    assert int(first.headers["X-Total-Count"]) >= 100
    assert len(first.json()) == 10

    second = client.get("/api/v1/approvals", params={"page": 2, "page_size": 10}, headers=auth(admin))
    assert len(second.json()) == 10
    first_ids = {item["id"] for item in first.json()}
    second_ids = {item["id"] for item in second.json()}
    assert first_ids.isdisjoint(second_ids)
    # 第一页应当比第二页更新
    assert min(item["created_at"] for item in first.json()) >= min(item["created_at"] for item in second.json())

    for params in ({"page": 0}, {"page_size": 0}, {"page_size": 500}):
        assert client.get("/api/v1/approvals", params=params, headers=auth(admin)).status_code == 422


def test_department_visibility_supports_multi_department_and_subtree():
    """部门可见性：all / 单部门 / 多部门 / 子树通配。"""
    from app.agent import department_visible

    assert department_visible("all", "any-department") is True
    assert department_visible("engineering", "engineering") is True
    assert department_visible("engineering", "finance") is False
    # 多部门（含空格容错）
    assert department_visible("engineering,it", "it") is True
    assert department_visible("engineering, it", "engineering") is True
    assert department_visible("engineering,it", "hr") is False
    # 子树：父部门文档对子部门可见，反向不可
    assert department_visible("engineering/*", "engineering/backend") is True
    assert department_visible("engineering/*", "engineering") is True
    assert department_visible("engineering/*", "finance") is False
    assert department_visible("engineering/backend", "engineering") is False


def test_multi_department_document_visible_to_each_listed_department():
    """多部门文档应被每个列出的部门检索到，且不被未列出的部门检索到。"""
    from app.agent import search_knowledge
    from app.db import Chunk, Document, SessionLocal, User
    from app.main import make_chunks

    db = SessionLocal()
    try:
        admin = db.query(User).filter_by(email="admin@example.com").one()
        body = "联合演练规范由工程与运维共同执行，演练前须提交方案并完成审批，演练后归档复盘记录。"
        document = Document(
            title="跨部门联合演练规范",
            department="engineering,ops",
            sensitivity="INTERNAL",
            body=body,
            created_by=admin.id,
        )
        db.add(document)
        db.flush()
        # 检索走的是 Chunk JOIN Document，没有切片的文档检索不到。
        db.add_all(
            Chunk(document_id=document.id, ordinal=index, content=chunk)
            for index, chunk in enumerate(make_chunks(body))
        )
        db.commit()
        try:
            engineer = db.query(User).filter_by(email="employee@example.com").one()
            operator = db.query(User).filter_by(email="ops@example.com").one()
            hr_user = db.query(User).filter_by(email="hr@example.com").one()
            title = "跨部门联合演练规范"
            assert title in [hit.title for hit in search_knowledge(db, engineer, "联合演练规范")]
            assert title in [hit.title for hit in search_knowledge(db, operator, "联合演练规范")]
            assert title not in [hit.title for hit in search_knowledge(db, hr_user, "联合演练规范")]
        finally:
            db.delete(document)
            db.commit()
    finally:
        db.close()


def test_schema_matches_models_in_test_database():
    """测试库应与模型定义一致，漂移守卫应判定为无差异。"""
    from app.main import detect_schema_drift

    assert detect_schema_drift() == []


def test_eval_dataset_is_wired_into_regression():
    """让 data/evals.jsonl 接入自动化回归，不再是没有消费方的摆设。"""
    import json
    from pathlib import Path

    from app.agent import RANK, department_visible, risk_action, search_knowledge
    from app.db import Document, SessionLocal, User

    path = Path(__file__).resolve().parents[1] / "data" / "evals.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(cases) >= 17, "评测集条目明显减少，请确认是否被误删"

    db = SessionLocal()
    try:
        users = {user.email: user for user in db.query(User).all()}
        for case in cases:
            assert case["user"] in users, f"{case['id']} 引用了未知账号"
            assert case["tags"], f"{case['id']} 缺少标签，未接入断言"
            tags = set(case["tags"])
            if "human-in-the-loop" in tags:
                assert risk_action(case["input"]) is not None, f"{case['id']} 应触发审批前置"
                continue
            # 其余用例都是「问制度」，不应被误判为高风险操作。
            assert risk_action(case["input"]) is None, f"{case['id']} 被误判为高风险操作"
            # 无论问什么，返回的每条引用都必须在该用户的授权范围内。
            user = users[case["user"]]
            for hit in search_knowledge(db, user, case["input"]):
                document = db.get(Document, hit.document_id)
                assert RANK[document.sensitivity] <= RANK[user.clearance], f"{case['id']} 越密级"
                assert department_visible(document.department, user.department), f"{case['id']} 越部门"
    finally:
        db.close()


def test_eval_dataset_expected_citations():
    """对评测集中写明引用期望的用例，逐条核对检索结果。

    混合检索引入语义召回后，「无 ground truth」的用例（如 access-001）不再保证
    返回空列表 —— 它会召回若干**有权阅读**的近似文档。这些用例的真实不变量是
    「不得泄露越权文档」，因此改用「禁止出现」断言而不是「必须为空」。
    """
    import json
    from pathlib import Path

    from app.agent import search_knowledge
    from app.db import SessionLocal, User

    expected = {
        "access-002": ["2026 差旅报销制度"],
        "policy-001": ["远程办公制度"],
        "rag-002": ["员工入职与试用期管理办法"],
        "rag-003": ["备份与恢复演练规程"],
        "rag-004": ["网络与 VPN 接入管理"],
    }
    # 用户无权阅读的文档绝不能出现在引用里，哪怕语义上高度相近。
    forbidden = {
        "access-001": ["2026 差旅报销制度"],
        "access-003": ["2026 差旅报销制度"],
        "access-005": ["薪酬结构与调薪机制"],
        "access-006": ["合同评审与用印管理办法"],
        "injection-001": ["2026 差旅报销制度"],
    }
    path = Path(__file__).resolve().parents[1] / "data" / "evals.jsonl"
    cases = {
        case["id"]: case
        for case in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }

    db = SessionLocal()
    try:
        users = {user.email: user for user in db.query(User).all()}
        for case_id, titles in expected.items():
            case = cases[case_id]
            got = [hit.title for hit in search_knowledge(db, users[case["user"]], case["input"])]
            for title in titles:
                assert title in got, f"{case_id} 期望引用 {title}，实际 {got}"
        for case_id, titles in forbidden.items():
            case = cases[case_id]
            got = [hit.title for hit in search_knowledge(db, users[case["user"]], case["input"])]
            for title in titles:
                assert title not in got, f"{case_id} 泄露了无权阅读的文档 {title}，实际 {got}"
    finally:
        db.close()


def test_known_departments_do_not_drift_from_seed_data():
    """部门白名单必须覆盖演示账号用到的全部部门，否则 seed 会写出非法数据。"""
    from app.departments import KNOWN_DEPARTMENTS
    from app.seed_data import DEMO_DOCUMENTS, DEMO_USERS

    seeded = {user[2] for user in DEMO_USERS} | {
        part.strip() for _, department, _, _ in DEMO_DOCUMENTS for part in department.split(",")
    }
    assert seeded - {"all"} <= set(KNOWN_DEPARTMENTS), f"白名单缺少：{sorted(seeded - set(KNOWN_DEPARTMENTS))}"


def test_chunk_text_clamps_overlap_and_never_silently_drops_content():
    """P0 回归：旧分块实现在 overlap >= size 时会静默返回空列表或直接抛 ValueError。"""
    import pytest

    from app.chunking import chunk_text

    text = "。".join(f"第{index}条：本条规定了相应的管理要求" for index in range(1, 121))

    with pytest.raises(ValueError):
        chunk_text(text, size=0)

    for size, overlap in ((350, 60), (100, 99), (100, 100), (100, 200)):
        chunks = chunk_text(text, size=size, overlap=overlap)
        assert chunks, f"size={size} overlap={overlap} 静默返回了空列表"
        assert all(len(chunk) <= size for chunk in chunks), f"size={size} 存在超长片段"
        joined = "".join(chunks)
        # 首尾内容都必须还在，证明没有整段丢失。
        assert "第1条" in joined
        assert "第120条" in joined


def test_documents_endpoint_rejects_unknown_department():
    """P0 回归：部门名拼错必须 422，而不是入库成功却对所有人不可见。"""
    finance = login("finance@example.com")
    payload = {
        "title": "审计用文档",
        "body": "这篇文档用于验证部门字段校验，正文长度需要满足最小要求。",
        "department": "enginering",
        "sensitivity": "INTERNAL",
    }
    response = client.post("/api/v1/documents", json=payload, headers=auth(finance))
    assert response.status_code == 422
    assert "未知部门" in response.text


def test_documents_endpoint_accepts_known_and_multi_department():
    """合法部门写法都应入库成功，并且新文档立刻可被对应部门检索到。"""
    from app.agent import search_knowledge
    from app.db import Document, SessionLocal, User

    finance = login("finance@example.com")
    created: list[int] = []
    try:
        for department in ("all", "it", "engineering,it", "engineering/*"):
            payload = {
                "title": f"跨部门联调规范 {department}",
                "body": "联调规范要求各团队在变更窗口内完成验证，并归档结果，正文长度满足最小要求。",
                "department": department,
                "sensitivity": "INTERNAL",
            }
            response = client.post("/api/v1/documents", json=payload, headers=auth(finance))
            assert response.status_code == 201, response.text
            body = response.json()
            assert body["chunks"] >= 1
            created.append(body["id"])

        # 入库后应立即重建向量，确保新文档马上可检索（不能等到下次全量重建）。
        with SessionLocal() as db:
            operator = db.query(User).filter_by(email="ops@example.com").one()
            titles = [hit.title for hit in search_knowledge(db, operator, "跨部门联调规范")]
            assert any("跨部门联调规范" in title for title in titles), titles
    finally:
        with SessionLocal() as db:
            for document_id in created:
                document = db.get(Document, document_id)
                if document is not None:
                    db.delete(document)
            db.commit()


def test_documents_endpoint_requires_finance_or_admin():
    employee = login("employee@example.com")
    payload = {
        "title": "越权入库尝试",
        "body": "普通员工不应该能写入知识库，这篇文档用于验证角色校验的正文长度。",
        "department": "all",
        "sensitivity": "PUBLIC",
    }
    assert client.post("/api/v1/documents", json=payload, headers=auth(employee)).status_code == 403


def test_retrieval_stats_requires_admin_and_hides_document_text():
    finance = login("finance@example.com")
    admin = login("admin@example.com")

    assert client.get("/api/v1/retrieval/stats", headers=auth(finance)).status_code == 403

    response = client.get("/api/v1/retrieval/stats", headers=auth(admin))
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] in {"faiss", "numpy", "pgvector"}
    assert "stored_vectors" in body
    assert "chunks" in body
    # 统计接口只返回汇总数字，绝不下发文档正文或标题。
    assert "documents" not in body
    assert "document" not in response.text.lower().replace("documents", "")


def test_labeled_retrieval_evaluation_meets_quality_gate():
    """带标注的检索评测：只有声明 relevant_titles 的用例进入召回分母。

    这条同时是**发布质量闸门**：每个有标注的文档都必须落在其授权用户的 Top-K 里。
    """
    finance = login("finance@example.com")
    admin = login("admin@example.com")

    assert client.post("/api/v1/retrieval/evaluations", json={"top_k": 4}, headers=auth(finance)).status_code == 403

    response = client.post("/api/v1/retrieval/evaluations", json={"top_k": 4}, headers=auth(admin))
    assert response.status_code == 201
    run = response.json()
    assert run["total_cases"] == 19
    # 19 条用例中有 7 条声明了 relevant_titles，其余是安全/审批用例，不进入召回分母。
    assert run["evaluated_cases"] == 7
    assert 0 <= run["recall_at_k"] <= 1
    assert 0 <= run["hit_rate_at_k"] <= 1
    assert run["pass_rate_at_k"] == 1.0, f"检索质量未达标：{run}"
    assert len(run["results"]) == run["total_cases"]

    history = client.get("/api/v1/retrieval/evaluations", headers=auth(admin))
    assert history.status_code == 200
    assert any(item["id"] == run["id"] for item in history.json())


def test_retrieval_evaluation_history_limit_is_bounded():
    admin = login("admin@example.com")
    assert client.get("/api/v1/retrieval/evaluations", params={"limit": 5}, headers=auth(admin)).status_code == 200
    assert client.get("/api/v1/retrieval/evaluations", params={"limit": 0}, headers=auth(admin)).status_code == 422
    assert client.get("/api/v1/retrieval/evaluations", params={"limit": -1}, headers=auth(admin)).status_code == 422


def test_approval_list_supports_status_and_keyword_filters():
    """筛选必须在服务端执行，且「待处理」统计保持全量口径。"""
    admin = login("admin@example.com")
    everything = client.get("/api/v1/approvals", params={"page_size": 100}, headers=auth(admin))
    assert everything.status_code == 200
    total = int(everything.headers["X-Total-Count"])
    assert total >= 100

    pending = client.get("/api/v1/approvals", params={"status": "PENDING", "page_size": 100}, headers=auth(admin))
    assert pending.status_code == 200
    assert pending.json(), "种子数据里应当有待处理审批单"
    assert all(item["status"] == "PENDING" for item in pending.json())
    assert int(pending.headers["X-Total-Count"]) < total
    # 待处理总数是待办计数，不随筛选变化。
    assert pending.headers["X-Total-Pending"] == everything.headers["X-Total-Pending"]

    rejected = client.get("/api/v1/approvals", params={"status": "REJECTED", "page_size": 100}, headers=auth(admin))
    assert all(item["status"] == "REJECTED" for item in rejected.json())

    keyword = client.get("/api/v1/approvals", params={"keyword": "客户联系人", "page_size": 100}, headers=auth(admin))
    assert keyword.status_code == 200
    assert keyword.json(), "关键词应能命中种子数据"
    for item in keyword.json():
        assert "客户联系人" in item["payload"]["message"]

    # 非法状态值应被参数校验拒绝。
    assert client.get("/api/v1/approvals", params={"status": "UNKNOWN"}, headers=auth(admin)).status_code == 422


def test_login_is_rate_limited_after_repeated_failures():
    """连续登录失败达到阈值后返回 429，避免固定演示口令被在线爆破。"""
    from app import main

    main._login_failures.clear()
    try:
        for _ in range(main.LOGIN_MAX_FAILURES):
            response = client.post(
                "/api/v1/auth/login", json={"email": "employee@example.com", "password": "wrong-password"}
            )
            assert response.status_code == 401

        blocked = client.post(
            "/api/v1/auth/login", json={"email": "employee@example.com", "password": "wrong-password"}
        )
        assert blocked.status_code == 429
        # 进入封禁窗口后，即使口令正确也应被拦下。
        assert client.post(
            "/api/v1/auth/login", json={"email": "employee@example.com", "password": "demo-password"}
        ).status_code == 429
    finally:
        main._login_failures.clear()

    # 窗口清空后恢复正常。
    assert client.post(
        "/api/v1/auth/login", json={"email": "employee@example.com", "password": "demo-password"}
    ).status_code == 200

