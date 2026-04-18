"""Cases for `first_sentence` projection (bench query sampler).

`first_sentence` takes a raw docstring (comment markers and all)
and returns the first-sentence query string, or `None` when the
docstring is too short, too noisy, or boilerplate.

Tags:

* `accepted`  - docstring yields a non-None query string; case
                carries the expected text.
* `rejected`  - docstring yields `None`; case carries a short
                label for test IDs only.

Secondary tags classify the scenario the same way
`case_docstrings.py` does (canonical, edge_case, unconventional,
invalid) so behaviour slices match language-coverage language.
"""

from __future__ import annotations

from pytest_cases import case

# ── Accepted ─────────────────────────────────────────────────────────


@case(tags=["accepted", "canonical"])
def case_python_triple_quoted() -> tuple[str, str]:
    raw = '"""Return a friendly greeting."""'
    return raw, "Return a friendly greeting."


@case(tags=["accepted", "canonical"])
def case_rust_triple_slash_run() -> tuple[str, str]:
    raw = "/// Construct an Error from the output of a failed command."
    return raw, "Construct an Error from the output of a failed command."


@case(tags=["accepted", "canonical"])
def case_go_line_comment() -> tuple[str, str]:
    raw = "// Greet says hello to the user."
    return raw, "Greet says hello to the user."


@case(tags=["accepted", "canonical"])
def case_ruby_hash_comment() -> tuple[str, str]:
    raw = "# Greet the user with a friendly message."
    return raw, "Greet the user with a friendly message."


@case(tags=["accepted", "canonical"])
def case_jsdoc_block() -> tuple[str, str]:
    raw = "/**\n * Check if an error is retryable (rate limit, server error, network error).\n */"
    return (
        raw,
        "Check if an error is retryable (rate limit, server error, network error).",
    )


@case(tags=["accepted", "edge_case"])
def case_jsdoc_multi_line_joined() -> tuple[str, str]:
    """Multi-line JSDoc with `@` tags on subsequent lines - the
    first sentence is everything before the first `.!?`; the tag
    lines do not leak in.
    """
    raw = "/**\n * Compute a hash.\n *\n * @param data input\n * @return hex string\n */"
    return raw, "Compute a hash."


@case(tags=["accepted", "edge_case"])
def case_python_multiline_summary() -> tuple[str, str]:
    """Summary on its own line, body in paragraph below."""
    raw = '"""Add two numbers.\n\n    The summary is one line; the body elaborates.\n    """'
    return raw, "Add two numbers."


@case(tags=["accepted", "unconventional"])
def case_rust_non_doc_style_comments() -> tuple[str, str]:
    """Plain `//` (not `///`) - rbtr attaches them, so the bench
    picks them up as queries too.
    """
    raw = "// Deliberately unconventional opening."
    return raw, "Deliberately unconventional opening."


# ── Rejected ────────────────────────────────────────────────────────


@case(tags=["rejected", "invalid"])
def case_too_short() -> str:
    """One-word docstring, below min length."""
    return '"""hi"""'


@case(tags=["rejected", "invalid"])
def case_todo_prefix() -> str:
    """Boilerplate `TODO` marker at start."""
    return '"""TODO: actually write this."""'


@case(tags=["rejected", "invalid"])
def case_fixme_prefix_in_comment() -> str:
    return "// FIXME: rewrite this broken thing."


@case(tags=["rejected", "invalid"])
def case_deprecated_prefix() -> str:
    return '"""Deprecated. Use Foo instead."""'


@case(tags=["rejected", "invalid"])
def case_jsdoc_tag_only() -> str:
    """A JSDoc block that starts with a `@param` tag - no human\n    prose, nothing useful to search with.\n"""
    return "/** @param data the input bytes */"


@case(tags=["rejected", "invalid"])
def case_empty_docstring() -> str:
    return '""""""'
