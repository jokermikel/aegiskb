import os

# 测试必须使用独立数据库，绝不能污染演示数据库。
os.environ["DATABASE_URL"] = "sqlite:///test_aegiskb.db"
os.environ["JWT_SECRET"] = "test-secret"
# 显式覆盖 .env 中可能存在的真实百炼密钥，保证测试只走本地模式。
os.environ["DASHSCOPE_API_KEY"] = ""

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

    def fake_run_agent(db, user, message, api_key=None):
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
