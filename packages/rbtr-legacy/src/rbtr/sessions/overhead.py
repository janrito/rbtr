"""Overhead data models for the `data_json` column.

Background LLM calls (compaction summaries, fact extraction)
incur costs outside the main conversation.  These are tracked
as overhead fragments with typed `data_json` payloads.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class CompactionTrigger(StrEnum):
    """What initiated the compaction."""

    MID_TURN = "mid-turn"
    AUTO_POST_TURN = "auto-post-turn"
    AUTO_OVERFLOW = "auto-overflow"
    MANUAL = "manual"


class CompactionOverhead(BaseModel):
    """`data_json` for `FragmentKind.OVERHEAD_COMPACTION`."""

    trigger: CompactionTrigger
    old_messages: int
    kept_messages: int
    summary_tokens: int
    model_name: str | None = None


class FactExtractionSource(StrEnum):
    """What triggered the fact extraction."""

    COMPACTION = "compaction"
    POST = "post"
    COMMAND = "command"


class FactExtractionOverhead(BaseModel):
    """`data_json` for `FragmentKind.OVERHEAD_FACT_EXTRACTION`."""

    source: FactExtractionSource
    added: int = 0
    confirmed: int = 0
    superseded: int = 0
    model_name: str | None = None
    fact_ids: list[str] = []


Overhead = CompactionOverhead | FactExtractionOverhead
"""Union of all overhead types for `save_overhead`."""
