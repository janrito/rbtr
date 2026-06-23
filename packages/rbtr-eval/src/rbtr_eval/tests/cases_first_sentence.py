"""Cases for the `first_sentence` projection.

`first_sentence` takes a raw docstring (comment markers and all) and
returns the first-sentence query string, or `None` when the docstring
is too short, too noisy, or boilerplate.

rbtr-eval deliberately does NOT strip comment markers from the
projected query — that would mean re-implementing per-language comment
syntax — so the markers appear in the output verbatim.

A representative scenario per behaviour; redundant per-language repeats
were removed.

Tags:

* `accepted` — yields a non-None query string; case carries the
               expected text.
* `rejected` — yields `None`; case carries a short label for test IDs.
"""

from __future__ import annotations

from pytest_cases import case

# ── Accepted ────────────────────────────────────────────────────────


@case(tags=["accepted"])
def case_python_triple_quoted() -> tuple[str, str]:
    """Single-line docstring: returned verbatim, markers and all."""
    raw = '"""Return a friendly greeting."""'
    expected = '"""Return a friendly greeting."""'
    return raw, expected


@case(tags=["accepted"])
def case_line_comment() -> tuple[str, str]:
    """A line-comment marker (`///`) is preserved, not stripped."""
    raw = "/// Construct an Error from the output of a failed command."
    expected = "/// Construct an Error from the output of a failed command."
    return raw, expected


@case(tags=["accepted"])
def case_jsdoc_block_first_sentence() -> tuple[str, str]:
    """Multi-line JSDoc: stops at the first sentence; gutter lines stay.

    The first sentence ends at the first `.`/`!`/`?` followed by
    whitespace, and we do not strip the `*` gutter, so it stays
    attached. The `@param` tag below is dropped.
    """
    raw = """\
/**
 * Compute a hash.
 *
 * @param data input
 */"""
    expected = "/**\n * Compute a hash."
    return raw, expected


# ── Rejected ────────────────────────────────────────────────────────


@case(tags=["rejected"])
def case_too_short() -> str:
    """Below `_QUERY_MIN_LEN`."""
    return '"""hi"""'


@case(tags=["rejected"])
def case_todo_prefix() -> str:
    """Boilerplate TODO marker after stripping leading noise."""
    return '"""TODO: actually write this."""'


@case(tags=["rejected"])
def case_jsdoc_tag_only() -> str:
    """JSDoc with only a `@param` tag — no prose sentence to extract."""
    return "/** @param data the input bytes */"


@case(tags=["rejected"])
def case_empty_docstring() -> str:
    """Empty triple-quote string."""
    return '""""""'
