from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint(monkeypatch) -> None:
    monkeypatch.setattr("app.main.vector_store.get_collection", lambda: None)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
