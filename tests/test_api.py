import os

os.environ["DATABASE_URL"] = "sqlite:///./test_aegiskb.db"
os.environ["JWT_SECRET"] = "test-secret"
# 环境变量优先于 .env；显式置空，防止本机真实密钥让回归测试访问外部模型。
os.environ["DASHSCOPE_API_KEY"] = ""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def setup_module():
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
    assert employee_response["mode"] == "local"
    assert finance_response["mode"] == "local"
    assert not employee_response["citations"]
    assert any(item["title"] == "2026 差旅报销制度" for item in finance_response["citations"])


def test_risky_request_creates_approval_and_only_admin_can_decide():
    employee = login("employee@example.com")
    admin = login("admin@example.com")
    result = client.post("/api/v1/agent/chat", json={"message": "请导出全部客户数据"}, headers=auth(employee)).json()
    assert result["requires_approval"] is True
    approval_id = result["approval_id"]
    forbidden = client.post(f"/api/v1/approvals/{approval_id}/decision", json={"decision": "APPROVED", "comment": "try"}, headers=auth(employee))
    assert forbidden.status_code == 403
    approved = client.post(f"/api/v1/approvals/{approval_id}/decision", json={"decision": "APPROVED", "comment": "security review complete"}, headers=auth(admin))
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"


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
    monkeypatch.setattr(main, "test_qwen_connection", lambda api_key: (api_key == "sk-valid-demo", "连接成功，API Key 可用。"))
    response = client.post(
        "/api/v1/settings/qwen/test",
        json={"api_key": "sk-valid-demo"},
        headers=auth(finance),
    )
    assert response.status_code == 200
    assert response.json()["connected"] is True
    assert response.json()["model"] == "qwen3.8-max"


def test_retrieval_stats_are_admin_only_and_do_not_expose_documents():
    finance = login("finance@example.com")
    admin = login("admin@example.com")
    assert client.get("/api/v1/retrieval/stats", headers=auth(finance)).status_code == 403

    response = client.get("/api/v1/retrieval/stats", headers=auth(admin))
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] in {"faiss", "numpy", "pgvector"}
    assert "stored_vectors" in body
    assert "chunks" in body
    assert "documents" not in body


def test_labeled_retrieval_evaluation_records_recall_metrics_and_meets_quality_gate():
    finance = login("finance@example.com")
    admin = login("admin@example.com")
    assert client.post("/api/v1/retrieval/evaluations", json={"top_k": 4}, headers=auth(finance)).status_code == 403

    response = client.post("/api/v1/retrieval/evaluations", json={"top_k": 4}, headers=auth(admin))
    assert response.status_code == 201
    run = response.json()
    assert run["total_cases"] >= run["evaluated_cases"] == 4
    assert run["top_k"] == 4
    assert 0 <= run["recall_at_k"] <= 1
    assert 0 <= run["hit_rate_at_k"] <= 1
    # The checked-in offline baseline is a release gate: each annotated
    # document must be in Top-4 for its authorized test user.
    assert run["pass_rate_at_k"] == 1.0
    assert 0 <= run["mrr"] <= 1
    assert len(run["results"]) == run["total_cases"]

    history = client.get("/api/v1/retrieval/evaluations", headers=auth(admin))
    assert history.status_code == 200
    assert any(item["id"] == run["id"] for item in history.json())
