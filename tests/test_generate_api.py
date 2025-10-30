from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient


def test_generate_routes_cover_intents(client_factory) -> None:
    client, _ = client_factory(
        ["evaluate_resume", "recommend_job", "mock_interview", "normal_chat"]
    )
    pdf_path = Path("examples/WANGSHIBO_CV.pdf")
    assert pdf_path.exists(), "Expected sample PDF for testing."

    with client as app_client:  # type: TestClient
        with pdf_path.open("rb") as pdf_file:
            resume_resp = app_client.post(
                "/generate",
                data={
                    "input": "请帮我评估这份简历。",
                    "web_search": "false",
                    "return_stream": "false",
                },
                files={"file": ("WANGSHIBO_CV.pdf", pdf_file, "application/pdf")},
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
                "input": "推荐一些机器学习工程师的岗位。",
                "web_search": "false",
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
                "input": "我们来模拟一次技术面试好吗？",
                "web_search": "false",
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
                "input": "谢谢，我们之后再联系。",
                "web_search": "false",
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
