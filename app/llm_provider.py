"""LLM provider abstraction supporting chat and embeddings."""

from __future__ import annotations

from typing import Iterable, List, Protocol

from openai import OpenAI

from .config import get_settings


class ChatProvider(Protocol):
    """Protocol for chat-based language models."""

    def chat(self, prompt: str) -> str:
        """Return the model response to the provided prompt."""

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Return embeddings for the given texts."""


class OpenAIProvider:
    """Default provider using OpenAI's API."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required.")
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.model_name
        self._temperature = settings.temperature

    def chat(self, prompt: str) -> str:
        """Send a single-shot prompt and return the model reply."""

        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for a job search and career coach demo."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or ""

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Generate embeddings for the provided texts."""

        texts_list = list(texts)
        if not texts_list:
            return []
        response = self._client.embeddings.create(
            model="text-embedding-3-small", input=texts_list
        )
        return [item.embedding for item in response.data]


def get_llm_provider() -> ChatProvider:
    """Return the configured LLM provider instance."""

    return OpenAIProvider()

