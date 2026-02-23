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
    """Persist unsaved messages from the current session.

    Compares ``session.message_history`` length against
    ``session.saved_count`` and inserts only the new tail.

    Parameters
    ----------
    run_cost:
        USD cost of the LLM run — attributed to the last
        ``ModelResponse`` in the new messages (if any).
    """
    session = engine.session
    history = session.message_history
    saved = session.saved_count

    if saved >= len(history):
        return

    new_messages = history[saved:]
    ctx = SessionContext(
        session_id=session.session_id,
        session_label=session.session_label,
        repo_owner=session.owner,
        repo_name=session.repo_name,
        model_name=session.model_name,
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
        session.saved_count = len(history)
    except OSError:
        log.warning("sessions: failed to persist %d messages", len(rows), exc_info=True)
