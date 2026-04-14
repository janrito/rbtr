"""Lightweight type aliases shared across the LLM and engine layers.

This module must stay free of heavy imports (`pydantic_ai`,
`duckdb`, `pyarrow`, `SessionStore`, etc.) so that both
`engine/core.py` and `llm/context.py` can import from it
without triggering expensive dependency chains.
"""

from __future__ import annotations

import anyio

#: Mutable slot shared between `Engine` and `LLMContext` so the
#: UI thread can signal an `anyio.Event` created inside the async
#: cancel watcher.  A 1-element list acts as a thread-safe pointer.
type CancelSlot = list[anyio.Event | None]
