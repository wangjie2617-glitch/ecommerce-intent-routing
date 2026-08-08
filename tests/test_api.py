import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import model_service


@pytest.fixture(autouse=True)
def skip_real_model_startup(monkeypatch):
    monkeypatch.setattr(model_service, "load", lambda: None)


def fake_prediction(texts: list[str]) -> list[dict[str, object]]:
    return [
        {
            "text": text,
            "intents": [{"label": "物流异常", "score": 0.91}],
            "route": {
                "department": "物流客服",
                "priority": "medium",
                "sla_minutes": 20,
                "manual_review": False,
                "reasons": ["测试替身"],
            },
            "model_type": "test-double",
            "latency_ms": 1.2,
        }
        for text in texts
    ]


def test_single_prediction_contract(monkeypatch) -> None:
    monkeypatch.setattr(model_service, "predict", fake_prediction)
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"text": "快递还没到"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intents"][0]["label"] == "物流异常"
    assert payload["route"]["department"] == "物流客服"


def test_batch_prediction_contract(monkeypatch) -> None:
    monkeypatch.setattr(model_service, "predict", fake_prediction)
    with TestClient(app) as client:
        response = client.post("/api/v1/predict/batch", json={"texts": ["快递没到", "物流没更新"]})
    assert response.status_code == 200
    assert response.json()["count"] == 2


def test_blank_text_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"text": "   "})
    assert response.status_code == 422


def test_demo_page_is_available() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "电商客户诉求智能路由" in response.text
