from typing import Iterable, List

from fastapi.testclient import TestClient

from app.main import app


class DummyLLM:
    def chat(self, prompt: str) -> str:
        return "Mock response with [0]"

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        return [[0.1] * 10 for _ in texts]


def test_generate_route(monkeypatch) -> None:
    client = TestClient(app)

    monkeypatch.setattr("app.main.get_llm_provider", lambda: DummyLLM())
    monkeypatch.setattr("app.main.intent_tools.intent_router", lambda _: "normal_chat")

    resp = client.post(
        "/generate",
        json={"input": "Hello there", "web_search": False, "return_stream": False},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["intent"] == "normal_chat"
    assert payload["text"]
    assert payload["tool_calls"] == ["normal_chat"]

