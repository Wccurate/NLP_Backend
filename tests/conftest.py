from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, List, Tuple

import pytest
from fastapi.testclient import TestClient

from app.main import app


class DummyLLM:
    """Deterministic substitute for the LLM provider used in tests."""

    def chat(self, prompt: str) -> str:
        if "Model Context Protocol" in prompt and "Respond with JSON" in prompt:
            return (
                '{"search": true, "queries": ["ai career market insights"],'
                ' "reason": "Check latest job market news."}'
            )
        if "Craft a concise, ideal answer" in prompt:
            return "An ideal candidate brings production NLP experience and collaboration."
        if "job matching assistant" in prompt:
            return "Consider the ML Engineer position [0] and the Data Scientist role [1]."
        if '"pros"' in prompt or "resume reviewer" in prompt:
            return (
                '{"pros":["Clear academic record"],'
                '"cons":["Need quantified impacts"],'
                '"suggestions":["Add measurable achievements"]}'
            )
        if "mock interview" in prompt:
            return (
                "Question: Describe a challenging project you led.\n"
                "Rubric:\n- Problem clarity\n- Depth of solution\n- Reflection on impact"
            )
        return "Thanks for the update! Keep me posted on your progress."

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[float((len(text) % 5) + 1)] * 4 for text in texts]


@pytest.fixture
def client_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> Callable[[List[str]], Tuple[TestClient, DummyLLM]]:
    """Prepare a TestClient with isolated storage and deterministic dependencies."""

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    db_path = data_dir / "app.db"
    if db_path.exists():
        db_path.unlink()

    chroma_dir = tmp_path / "chroma"
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    dummy_llm = DummyLLM()

    class DummySettings:
        model_name = "dummy-model"
        temperature = 0.0
        history_window = 10
        primary_intent_mode = "openai"
        openai_api_key = "dummy"
        primary_search_provider = "tavily"
        tavily_api_key = "dummy"
        tavily_endpoint = "https://api.tavily.com/search"

    DummySettings.chroma_dir = chroma_dir

    monkeypatch.setattr("app.main.get_settings", lambda: DummySettings())
    monkeypatch.setattr("app.main.get_llm_provider", lambda: dummy_llm)
    monkeypatch.setattr("app.main.vector_store.get_collection", lambda: None)

    def fake_add_texts(texts, metadatas=None, embedder=None):
        return [f"doc-{idx}" for idx, _ in enumerate(texts)]

    monkeypatch.setattr("app.main.vector_store.add_texts", fake_add_texts)
    monkeypatch.setattr("app.main.vector_store.delete", lambda ids: None)

    def fake_search(question, hyde_text=None, extra_corpus=None, embedder=None):
        return [
            {
                "id": "jobs_demo#42",
                "text": "ML Engineer role focusing on NLP systems.",
                "score": 0.87,
                "source": "jobs_demo#42",
            },
            {
                "id": "jobs_demo#17",
                "text": "Data Scientist role delivering insights to stakeholders.",
                "score": 0.81,
                "source": "jobs_demo#17",
            },
        ]

    monkeypatch.setattr("app.main.vector_store.search", fake_search)

    class DummySearch:
        def __init__(self) -> None:
            self.queries: List[str] = []

        def search(self, query: str, *, max_results: int = 5):
            self.queries.append(query)
            return [
                {
                    "title": "Mock Result",
                    "url": "https://example.com",
                    "snippet": "Recent industry highlights for testing.",
                }
            ]

    def factory(intent_sequence: List[str]) -> Tuple[TestClient, DummyLLM]:
        queue = list(intent_sequence)

        def fake_intent_router(_: str) -> str:
            if queue:
                return queue.pop(0)
            return "normal_chat"

        monkeypatch.setattr("app.main.intent_tools.intent_router", fake_intent_router)
        search_stub = DummySearch()
        monkeypatch.setattr("app.main.get_search_client", lambda: search_stub)
        client = TestClient(app)
        return client, dummy_llm

    def cleanup() -> None:
        if db_path.exists():
            db_path.unlink()
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)

    request.addfinalizer(cleanup)
    return factory
