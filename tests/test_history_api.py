from __future__ import annotations

import io

from fastapi.testclient import TestClient


SAMPLE_PDF_BYTES = bytes.fromhex(
    "255044462d312e340a25d0d4c5d80a312030206f626a0a3c3c2f54797065202f436174616c6f672f5061676573203220302052203e3e0a656e646f626a0a322030206f626a0a3c3c2f54797065202f506167652f506172656e742031203020522f4d65646961426f78205b30203020353030203730305d2f436f6e74656e7473203320302052203e3e0a656e646f626a0a332030206f626a0a3c3c2f4c656e6774682031372f46696c746572202f466c6174654465636f64653e3e73747265616d0a4254202f463120313220546620284a6f62205365656b65722044656d6f20526573756d65290a54202f46312031302054662028457870657269656e63656420536b696c6c7320616e6420486f6f6b73290a454f53747265616d0a656e646f626a0a7064660a747261696c65720a3c3c2f526f6f742031203020522f53697a6520343e3e0a7374617274787265660a302030200a454f460a"
)


def test_history_returns_combined_entries(client_factory) -> None:
    client, _ = client_factory(["evaluate_resume", "normal_chat"])

    with client as app_client:  # type: TestClient
        eval_resp = app_client.post(
            "/generate",
            data={
                "input": "Please review the attached resume.",
                "return_stream": "false",
            },
            files={"file": ("resume.pdf", io.BytesIO(SAMPLE_PDF_BYTES), "application/pdf")},
        )
        assert eval_resp.status_code == 200

        chat_resp = app_client.post(
            "/generate",
            data={
                "input": "By the way, what's on the agenda today?",
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
