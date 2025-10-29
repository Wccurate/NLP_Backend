from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def test_history_returns_combined_entries(client_factory) -> None:
    client, _ = client_factory(["evaluate_resume", "normal_chat"])
    pdf_path = Path("examples/WANGSHIBO_CV.pdf")
    assert pdf_path.exists()

    with client as app_client:  # type: TestClient
        with pdf_path.open("rb") as pdf:
            eval_resp = app_client.post(
                "/generate",
                data={
                    "input": "请评价一下我附带的简历。",
                    "web_search": "false",
                    "return_stream": "false",
                },
                files={"file": ("WANGSHIBO_CV.pdf", pdf, "application/pdf")},
            )
        assert eval_resp.status_code == 200

        chat_resp = app_client.post(
            "/generate",
            data={
                "input": "顺便问下，今天的安排如何？",
                "web_search": "false",
                "return_stream": "false",
            },
        )
        assert chat_resp.status_code == 200

        history_resp = app_client.get("/history", params={"limit": 10})
        assert history_resp.status_code == 200
        history = history_resp.json()

        assert len(history) >= 4
        assert history[0]["role"] == "user"
        assert "<document>" in history[0]["content"]
        intents = [entry["intent"] for entry in history]
        assert "evaluate_resume" in intents
        assert history[-1]["intent"] == "normal_chat"
        assert history[-1]["role"] == "assistant"
