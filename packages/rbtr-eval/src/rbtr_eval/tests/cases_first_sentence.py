"""Cases for `first_sentence` projection.

`first_sentence` takes a raw docstring (comment markers and
all) and returns the first-sentence query string, or `None`
when the docstring is too short, too noisy, or boilerplate.

rbtr-eval deliberately does NOT strip comment markers from the
projected query - that would mean re-implementing per-language
comment syntax.  Cases therefore expect the markers to appear
in the output verbatim.

Tags:

* `accepted`  - docstring yields a non-None query string; case
                carries the expected text.
* `rejected`  - docstring yields `None`; case carries a short
                label for test IDs only.
"""

from __future__ import annotations

from pytest_cases import case

# ── Accepted ────────────────────────────────────────────────────────


@case(tags=["accepted"])
def case_python_triple_quoted() -> tuple[str, str]:
    """Python single-line docstring."""
    raw = '"""Return a friendly greeting."""'
    expected = '"""Return a friendly greeting."""'
    return raw, expected


@case(tags=["accepted"])
def case_rust_triple_slash() -> tuple[str, str]:
    """Rust /// doc comment."""
    raw = "/// Construct an Error from the output of a failed command."
    expected = "/// Construct an Error from the output of a failed command."
    return raw, expected


@case(tags=["accepted"])
def case_go_line_comment() -> tuple[str, str]:
    """Go // comment."""
    raw = "// Greet says hello to the user."
    expected = "// Greet says hello to the user."
    return raw, expected


@case(tags=["accepted"])
def case_ruby_hash_comment() -> tuple[str, str]:
    """Ruby # comment."""
    raw = "# Greet the user with a friendly message."
    expected = "# Greet the user with a friendly message."
    return raw, expected


@case(tags=["accepted"])
def case_jsdoc_block_first_sentence() -> tuple[str, str]:
    """Multi-line JSDoc with @tag below.  The first sentence
    ends at the first `.`/`!`/`?` followed by whitespace; gutter
    lines stay attached because we do not strip them.
    """
    raw = """\
/**
 * Compute a hash.
 *
 * @param data input
 */"""
    expected = "/**\n * Compute a hash."
    return raw, expected


@case(tags=["accepted"])
def case_python_multiline_summary() -> tuple[str, str]:
    """Python summary line + body paragraph."""
    raw = '''\
"""Add two numbers carefully.

    The summary is one line; the body elaborates.
    """'''
    expected = '"""Add two numbers carefully.'
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
def case_fixme_prefix_in_comment() -> str:
    """FIXME after a // comment marker."""
    return "// FIXME: rewrite this broken thing."


@case(tags=["rejected"])
def case_deprecated_prefix() -> str:
    """Deprecated boilerplate."""
    return '"""Deprecated. Use Foo instead."""'


@case(tags=["rejected"])
def case_jsdoc_tag_only() -> str:
    """JSDoc that is nothing but a `@param` tag."""
    return "/** @param data the input bytes */"


@case(tags=["rejected"])
def case_empty_docstring() -> str:
    """Empty triple-quote string."""
    return '""""""'
