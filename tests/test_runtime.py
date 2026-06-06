"""Tests for cooperative pipeline cancellation."""
import threading

import pytest

from src.agent.runtime import PipelineCancelled, cancellation_context, cancellation_guard


def test_guard_stops_cancelled_pipeline_before_node():
    event = threading.Event()
    event.set()
    called = False

    def node(state):
        nonlocal called
        called = True
        return state

    with pytest.raises(PipelineCancelled), cancellation_context(event):
        cancellation_guard(node)({})

    assert called is False


def test_guard_stops_cancelled_pipeline_after_node():
    event = threading.Event()

    def node(state):
        event.set()
        return state

    with pytest.raises(PipelineCancelled), cancellation_context(event):
        cancellation_guard(node)({})
