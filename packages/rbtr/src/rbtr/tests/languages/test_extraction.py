"""Engine-level extraction tests.

Per-language extraction lives in each `rbtr-lang-*` package. These tests pin
the language-agnostic engine mechanics: the name/scope resolver fallbacks and
capture-kind filtering in `extract_symbols` (exercised with synthetic
registrations over the Python grammar), and the content-less host-presence
chunk emitted for every registered language.
"""

from __future__ import annotations

import pytest

from rbtr.git import FileEntry
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction
from rbtr.languages.treesitter import extract_symbols


@pytest.mark.parametrize("lang", sorted(get_manager().all_language_ids()))
def test_empty_source_yields_host_presence(lang: str) -> None:
    """Empty source yields one content-less host-presence chunk, for every
    registered language.

    Records the file's host language so the blob-dedup gate skips an empty
    file on later builds instead of re-parsing it every time.
    """
    chunks = extract_file(FileEntry("input", "sha1", b""), lang)
    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert chunks[0].language == lang


def test_anonymous_chunk_when_name_capture_missing() -> None:
    """Chunks get name='<anonymous>' when the query omits the name capture."""
    grammar = get_manager().load_grammar("python")
    assert grammar is not None
    query_no_name = "(function_definition) @function\n"
    src = b"""\
def hello():
    pass
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query_no_name))
    chunks = list(extract_symbols(reg, "test.py", "sha1", src, grammar))
    assert len(chunks) >= 1
    assert chunks[0].name == "<anonymous>"


def test_scope_extractor_owns_scope_address() -> None:
    """A `scope_extractor` overrides the default scope with its own segments."""
    grammar = get_manager().load_grammar("python")
    assert grammar is not None
    query = "(function_definition name: (identifier) @_fn_name) @function\n"
    src = b"def hello():\n    pass\n"
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query))
    reg.scope_extractor(lambda _resolver, _cap, _node, _caps: ["a", "b"])
    chunks = list(extract_symbols(reg, "test.py", "sha1", src, grammar))
    assert len(chunks) == 1
    assert chunks[0].scope == "a::b"


def test_unknown_capture_name_ignored() -> None:
    """Captures not in _CAPTURE_KINDS are silently skipped."""
    grammar = get_manager().load_grammar("python")
    assert grammar is not None
    query_unknown = """\
(function_definition
  name: (identifier) @_fn_name) @function
(class_definition) @unknown_thing
"""
    src = b"""\
def f():
    pass

class C:
    pass
"""
    reg = LanguageRegistration(id="faketest", extraction=QueryExtraction(query=query_unknown))
    chunks = list(extract_symbols(reg, "test.py", "sha1", src, grammar))
    kinds = {c.kind for c in chunks}
    assert "function" in kinds
    assert "class" not in kinds
