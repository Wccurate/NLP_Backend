"""Intent classification utilities."""

from __future__ import annotations

import json
import logging
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ..config import get_settings
from ..llm_provider import ChatProvider, get_llm_provider

logger = logging.getLogger(__name__)

INTENTS: List[str] = [
    "normal_chat",
    "mock_interview",
    "evaluate_resume",
    "recommend_job",
]

_SEED_DATA: List[Tuple[str, str]] = [
    ("Tell me about the work culture at startups.", "normal_chat"),
    ("Thanks, that's helpful!", "normal_chat"),
    ("Can you ask me a behavioral interview question?", "mock_interview"),
    ("Let's practice data scientist interviews.", "mock_interview"),
    ("Here is my resume, can you critique it?", "evaluate_resume"),
    ("What are the weak points in this CV?", "evaluate_resume"),
    ("Match me with AI job openings.", "recommend_job"),
    ("Find machine learning engineer roles in New York.", "recommend_job"),
]

_VECTORIZER: TfidfVectorizer | None = None
_CLASSIFIER: LogisticRegression | None = None


def _train_fallback() -> None:
    global _VECTORIZER, _CLASSIFIER
    texts, labels = zip(*_SEED_DATA)
    _VECTORIZER = TfidfVectorizer(ngram_range=(1, 2))
    X = _VECTORIZER.fit_transform(texts)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, labels)
    _CLASSIFIER = clf
    logger.info("Fallback intent classifier trained.")


def _ensure_fallback() -> None:
    if _VECTORIZER is None or _CLASSIFIER is None:
        _train_fallback()


def classify_openai(llm: ChatProvider, text: str) -> str:
    """Use the LLM to infer the intent label."""

    prompt = (
        "Classify the user's intent into one of the following labels:\n"
        f"{', '.join(INTENTS)}.\n"
        "Return ONLY the label.\n"
        "User text:\n"
        f"{text}"
    )
    raw = llm.chat(prompt).strip()
    label = raw.lower()
    if label not in INTENTS:
        logger.warning("LLM intent output invalid: %s", raw)
        raise ValueError("Invalid LLM intent response.")
    return label


def classify_fallback(text: str) -> str:
    """Fallback classifier using TF-IDF + Logistic Regression."""

    _ensure_fallback()
    assert _VECTORIZER is not None and _CLASSIFIER is not None
    X = _VECTORIZER.transform([text])
    label = _CLASSIFIER.predict(X)[0]
    return label


def intent_router(text: str) -> str:
    """Determine the intent using the configured primary mode."""

    settings = get_settings()
    primary = settings.primary_intent_mode.lower()
    llm: ChatProvider | None = None
    if primary == "openai":
        try:
            llm = get_llm_provider()
            label = classify_openai(llm, text)
            logger.info("Intent via OpenAI: %s", label)
            return label
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI intent classification failed: %s", exc)

    label = classify_fallback(text)
    logger.info("Intent via fallback: %s", label)
    return label


def serialize_seed_data() -> str:
    """Return the fallback dataset as JSON (debug helper)."""

    payload = [{"text": text, "label": label} for text, label in _SEED_DATA]
    return json.dumps(payload, indent=2)

