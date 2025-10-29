"""Helpers for maintaining a sliding window over conversation history."""

from __future__ import annotations

from typing import Iterable, List, MutableSequence, Sequence, TypeVar

T = TypeVar("T")


def last_k(messages: Sequence[T], k: int = 10) -> List[T]:
    """Return the last k items from the sequence."""

    if k <= 0:
        return []
    if len(messages) <= k:
        return list(messages)
    return list(messages[-k:])

