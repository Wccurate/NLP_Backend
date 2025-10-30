from __future__ import annotations

import json
import io

from fastapi.testclient import TestClient


SAMPLE_PDF_BYTES = bytes.fromhex(
    "255044462d312e340a25d0d4c5d80a312030206f626a0a3c3c2f54797065202f436174616c6f672f5061676573203220302052203e3e0a656e646f626a0a322030206f626a0a3c3c2f54797065202f506167652f506172656e742031203020522f4d65646961426f78205b30203020353030203730305d2f436f6e74656e7473203320302052203e3e0a656e646f626a0a332030206f626a0a3c3c2f4c656e6774682031372f46696c746572202f466c6174654465636f64653e3e73747265616d0a4254202f463120313220546620284a6f62205365656b65722044656d6f20526573756d65290a54202f46312031302054662028457870657269656e63656420536b696c6c7320616e6420486f6f6b73290a454f53747265616d0a656e646f626a0a7064660a747261696c65720a3c3c2f526f6f742031203020522f53697a6520343e3e0a7374617274787265660a302030200a454f460a"
)


def test_generate_routes_cover_intents(client_factory) -> None:
    client, _ = client_factory(
        ["evaluate_resume", "recommend_job", "mock_interview", "normal_chat"]
    )

    with client as app_client:  # type: TestClient
        resume_resp = app_client.post(
            "/generate",
            data={
                "input": "Please evaluate this resume.",
                "return_stream": "false",
            },
            files={"file": ("resume.pdf", io.BytesIO(SAMPLE_PDF_BYTES), "application/pdf")},
        )
        assert resume_resp.status_code == 200
        resume_payload = resume_resp.json()
        assert resume_payload["intent"] == "evaluate_resume"
        parsed_json = json.loads(resume_payload["text"])
        assert "pros" in parsed_json and "suggestions" in parsed_json
        assert resume_payload["tool_calls"] == ["evaluate_resume"]

        rag_resp = app_client.post(
            "/generate",
            data={
                "input": "Recommend some machine learning engineer roles.",
                "return_stream": "false",
            },
        )
        assert rag_resp.status_code == 200
        rag_payload = rag_resp.json()
        assert rag_payload["intent"] == "recommend_job"
        assert rag_payload["sources"]
        assert rag_payload["tool_calls"] == ["recommend_job"]
        assert "[0]" in rag_payload["text"]

        interview_resp = app_client.post(
            "/generate",
            data={
                "input": "Can we run a technical mock interview?",
                "return_stream": "false",
            },
        )
        assert interview_resp.status_code == 200
        interview_payload = interview_resp.json()
        assert interview_payload["intent"] == "mock_interview"
        assert "Question:" in interview_payload["text"]
        assert "Rubric" in interview_payload["text"]
        assert interview_payload["tool_calls"] == ["mock_interview"]

        chat_resp = app_client.post(
            "/generate",
            data={
                "input": "Thanks, let's catch up later.",
                "return_stream": "false",
            },
        )
        assert chat_resp.status_code == 200
        chat_payload = chat_resp.json()
        assert chat_payload["intent"] == "normal_chat"
        assert "Thanks" in chat_payload["text"]
        assert chat_payload["tool_calls"] == ["normal_chat"]


def test_add_job_description_endpoint(client_factory) -> None:
    client, _ = client_factory([])

    with client as app_client:  # type: TestClient
        resp = app_client.post(
            "/jobs",
            json={
                "text": "Senior NLP Engineer responsible for deploying language models.",
                "title": "Senior NLP Engineer",
                "metadata": {"location": "Remote"},
            },
        )

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["inserted"] > 0
    assert payload["ids"]
