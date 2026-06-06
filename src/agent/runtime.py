"""Per-request runtime controls for cancellation-aware LangGraph execution."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator


class PipelineCancelled(RuntimeError):
    """Raised between graph nodes after the HTTP request budget expires."""


_local = threading.local()


@contextmanager
def cancellation_context(
    cancel_event: threading.Event | None,
    deadline: float | None = None,
) -> Iterator[None]:
    previous = getattr(_local, "cancel_event", None)
    previous_deadline = getattr(_local, "deadline", None)
    _local.cancel_event = cancel_event
    _local.deadline = deadline
    try:
        check_cancelled()
        yield
    finally:
        _local.cancel_event = previous
        _local.deadline = previous_deadline


def check_cancelled() -> None:
    event = getattr(_local, "cancel_event", None)
    if event is not None and event.is_set():
        raise PipelineCancelled("Pipeline execution was cancelled.")


def remaining_seconds() -> float:
    deadline = getattr(_local, "deadline", None)
    if deadline is None:
        return float("inf")
    return max(0.0, deadline - time.monotonic())


def current_cancel_event() -> threading.Event | None:
    return getattr(_local, "cancel_event", None)


def cancellation_guard(node: Callable) -> Callable:
    """Check request cancellation immediately before and after a graph node."""

    @wraps(node)
    def guarded(state):
        check_cancelled()
        result = node(state)
        check_cancelled()
        return result

    return guarded
