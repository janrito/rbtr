"""Default docstring-extraction behaviour across languages.

Policy: rbtr extracts symbol-adjacent documentation into chunk
content by default for every supported language.  These tests
pin that policy down.

Observable behaviours exercised:

* `test_documented_chunk_includes_doc_text` — for every
  `documented` case, the snippet is present in the chunk's
  content after a default extraction.
* `test_stripping_removes_doc_text` — with
  `--strip-docstrings`, the same snippet is absent.
* `test_stripping_preserves_chunk_id` and
  `test_stripping_preserves_line_count` — stripping does not
  shift line numbers or chunk identity.
* `test_no_phantom_documentation` — for every `undocumented`
  case, the probe snippet does not appear.
* `test_empty_doc_types_suppresses_leading_attachment` and
  `test_attachment_shifts_line_start_for_exterior_plugins`
  — engine contract: forcing `doc_comment_node_types=frozenset()`
  disables sibling-walk attachment for exterior plugins and
  leaves interior (Python) extraction unaffected.

All test data lives in `case_docstrings.py`.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.index.models import Chunk
from rbtr.index.treesitter import extract_symbols
from rbtr.languages import get_manager

_CASES = "tests.languages.case_docstrings"


def _chunk_for(
    lang: str,
    source: str,
    symbol_name: str,
    *,
    strip: bool = False,
    override_doc_comment_node_types: frozenset[str] | None = None,
) -> Chunk:
    """Run production plugin config on *source* and return the
    chunk named *symbol_name*.

    Uses whatever the plugin actually registers — including
    `doc_comment_node_types` — so these tests exercise
    default-on behaviour.  Tests that probe the engine's
    fall-back behaviour can force `doc_comment_node_types` to an
    explicit value (typically `frozenset()` to suppress
    leading-comment attachment).
    """
    mgr = get_manager()
    grammar = mgr.load_grammar(lang)
    assert grammar is not None, f"grammar for {lang} not installed"
    reg = mgr.get_registration(lang)
    assert reg is not None
    assert reg.query is not None
    ext = next(iter(reg.extensions), ".txt")

    doc_types = (
        reg.doc_comment_node_types
        if override_doc_comment_node_types is None
        else override_doc_comment_node_types
    )

    chunks = extract_symbols(
        f"test{ext}",
        "sha1",
        source.encode(),
        grammar,
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
        doc_comment_node_types=doc_types,
        strip_docstrings=strip,
    )
    matches = [c for c in chunks if c.name == symbol_name]
    assert matches, (
        f"no chunk named {symbol_name!r} in {lang} source; extracted: {[c.name for c in chunks]}"
    )
    assert len(matches) == 1, (
        f"multiple chunks named {symbol_name!r} in {lang} source; "
        f"cases must keep names unique: {matches}"
    )
    return matches[0]


# ── Documented symbols ──────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=_CASES,
    has_tag="documented",
)
def test_documented_chunk_includes_doc_text(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """By default the chunk content carries the symbol's docs."""
    chunk = _chunk_for(lang, source, name)
    assert snippet in chunk.content, (
        f"expected {snippet!r} in {lang}.{name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=_CASES,
    has_tag="documented",
)
def test_stripping_removes_doc_text(lang: str, source: str, name: str, snippet: str) -> None:
    """`--strip-docstrings` blanks the documentation text."""
    chunk = _chunk_for(lang, source, name, strip=True)
    assert snippet not in chunk.content, (
        f"expected {snippet!r} to be absent from stripped "
        f"{lang}.{name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=_CASES,
    has_tag="documented",
)
def test_stripping_preserves_chunk_id(lang: str, source: str, name: str, snippet: str) -> None:
    """Stripping does not change `chunk.id`.

    `chunk_id` hashes `(file_path, name, line_start)`.  Stripping
    preserves `line_start` (docstring bytes are replaced with
    whitespace, not removed), so the ID must not shift.
    """
    kept = _chunk_for(lang, source, name, strip=False)
    stripped = _chunk_for(lang, source, name, strip=True)
    assert kept.id == stripped.id


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=_CASES,
    has_tag="documented",
)
def test_stripping_preserves_line_count(lang: str, source: str, name: str, snippet: str) -> None:
    """Stripping preserves line_start, line_end, and the newline
    count in chunk content.
    """
    kept = _chunk_for(lang, source, name, strip=False)
    stripped = _chunk_for(lang, source, name, strip=True)
    assert kept.line_start == stripped.line_start
    assert kept.line_end == stripped.line_end
    assert kept.content.count("\n") == stripped.content.count("\n")


# ── Undocumented symbols ────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=_CASES,
    has_tag="undocumented",
)
def test_no_phantom_documentation(lang: str, source: str, name: str, snippet: str) -> None:
    """Symbols without documentation do not gain any in content.

    For `invalid` cases, the snippet marks text that *looks*
    like documentation but must not be swept into the chunk.
    """
    chunk = _chunk_for(lang, source, name)
    assert snippet not in chunk.content, (
        f"unexpected {snippet!r} in {lang}.{name} content; got:\n{chunk.content!r}"
    )


# ── Engine contract expressed over the case set ─────────────────────


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=_CASES,
    has_tag="documented",
)
def test_empty_doc_types_suppresses_leading_attachment(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """Forcing `doc_comment_node_types=frozenset()` disables
    sibling-walk attachment.

    * Exterior plugins (Go, Rust, JS, …) rely on the walk —
      with the override the documentation snippet must disappear.
    * Interior plugins (Python) capture docstrings via
      `@_docstring` and are unaffected — the snippet remains.

    Replaces the hand-rolled Go-only plain test that previously
    asserted this invariant.
    """
    mgr = get_manager()
    reg = mgr.get_registration(lang)
    assert reg is not None

    chunk = _chunk_for(lang, source, name, override_doc_comment_node_types=frozenset())
    if reg.doc_comment_node_types:
        assert snippet not in chunk.content, (
            f"expected {snippet!r} to be absent from {lang}.{name} "
            f"when doc_comment_node_types is forced empty; "
            f"got:\n{chunk.content!r}"
        )
    else:
        assert snippet in chunk.content


@parametrize_with_cases(
    "lang, source, name, snippet",
    cases=_CASES,
    has_tag="documented",
)
def test_attachment_shifts_line_start_for_exterior_plugins(
    lang: str, source: str, name: str, snippet: str
) -> None:
    """When sibling-walk attachment fires, `line_start` moves up
    to cover the earliest attached comment and `chunk.id`
    recomputes accordingly.

    * Exterior plugins (default `doc_comment_node_types` set)
      → default `line_start` is earlier than the forced-empty
      extraction; ids differ.
    * Interior plugins (Python) → no sibling walk, `line_start`
      and id are identical to the forced-empty extraction.
    """
    mgr = get_manager()
    reg = mgr.get_registration(lang)
    assert reg is not None

    default = _chunk_for(lang, source, name)
    no_attach = _chunk_for(lang, source, name, override_doc_comment_node_types=frozenset())
    if reg.doc_comment_node_types:
        assert default.line_start < no_attach.line_start
        assert default.id != no_attach.id
    else:
        assert default.line_start == no_attach.line_start
        assert default.id == no_attach.id
