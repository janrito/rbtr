"""Auto-save new messages to the session store after each LLM turn."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelResponse

from rbtr.sessions.serialise import MessageRow, SessionContext, prepare_row

if TYPE_CHECKING:
    from .core import Engine

log = logging.getLogger(__name__)


def save_new_messages(engine: Engine, *, run_cost: float | None = None) -> None:
    """Persist unsaved messages from the current state.

    Compares ``state.message_history`` length against
    ``state.saved_count`` and inserts only the new tail.

    Parameters
    ----------
    run_cost:
        USD cost of the LLM run — attributed to the last
        ``ModelResponse`` in the new messages (if any).
    """
    state = engine.state
    history = state.message_history
    saved = state.saved_count

    if saved >= len(history):
        return

    new_messages = history[saved:]
    ctx = SessionContext(
        session_id=state.session_id,
        session_label=state.session_label,
        repo_owner=state.owner,
        repo_name=state.repo_name,
        model_name=state.model_name,
    )

    # Find the last ModelResponse index for cost attribution.
    last_response_idx: int | None = None
    if run_cost is not None:
        for i in range(len(new_messages) - 1, -1, -1):
            if isinstance(new_messages[i], ModelResponse):
                last_response_idx = i
                break

    rows: list[MessageRow] = []
    for i, msg in enumerate(new_messages):
        cost = run_cost if i == last_response_idx else None
        rows.append(
            prepare_row(
                msg,
                context=ctx,
                row_id=engine._store.new_id(),
                cost=cost,
            )
        )

    try:
        engine._store.save_messages(rows)
        state.saved_count = len(history)
    except OSError:
        log.warning("sessions: failed to persist %d messages", len(rows), exc_info=True)
