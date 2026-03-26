"""Operational prompts sent to the model during streaming.

Short, static messages injected by the streaming pipeline when
it needs the model to react to a runtime condition (token limit
hit, interrupted response, mid-turn compaction).  These are not
user-facing review instructions — those live in `rbtr.prompts`.
"""

# ── Tool-call limit ─────────────────────────────────────────────────

LIMIT_SUMMARY = """\
You have reached the tool-call limit for this turn.
Summarize what you accomplished so far and what remains to be done,
so the user can decide whether to ask you to continue."""

# ── Mid-turn compaction ─────────────────────────────────────────────

MID_TURN_COMPACTION = "The model is mid-turn with active tool calls."

# ── Interrupted responses ────────────────────────────────────────────

INTERRUPTED_TOOL_MESSAGES: dict[str, str] = {
    "length": """\
Your tool call was truncated because the output exceeded the token limit.
The tool was NOT executed.
Use a different strategy that produces smaller tool calls.""",
    "content_filter": """\
Your tool call was blocked by a content filter.
The tool was NOT executed.
Rephrase your content and retry.""",
    "error": """\
Your tool call was interrupted by a provider error.
The tool was NOT executed. Retry.""",
}

INTERRUPTED_TEXT_MESSAGES: dict[str, str] = {
    "length": """\
Your response was cut short by the output token limit.
Continue from where you stopped.""",
    "content_filter": """\
Your response was blocked by a content filter.
Rephrase and continue.""",
    "error": """\
Your response was interrupted by a provider error.
Continue from where you stopped.""",
}
