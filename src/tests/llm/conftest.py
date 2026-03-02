"""Shared fixtures for LLM pipeline tests."""

from __future__ import annotations

import queue
from collections.abc import Generator

import pytest

from rbtr.engine import Engine
from rbtr.llm.context import LLMContext
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState
from tests.conftest import drain, has_event_type, output_texts  # noqa: F401


@pytest.fixture
def engine() -> Generator[Engine]:
    """Default engine with auto-cleanup."""
    state = EngineState(owner="testowner", repo_name="testrepo")
    eng = Engine(state, queue.Queue(), store=SessionStore())
    yield eng
    eng.close()


@pytest.fixture
def ctx(engine: Engine) -> LLMContext:
    """LLMContext backed by the default engine fixture."""
    return engine._llm_context()
