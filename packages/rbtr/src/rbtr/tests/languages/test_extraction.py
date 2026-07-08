"""Tests for language extraction — symbols, imports, and properties.

All test data lives in `case_extraction.py`. Three behaviors:

1. **Symbol extraction** (`symbol` tag) — given source in
   language X, extraction produces chunks with correct
   `(kind, name, scope)`.
2. **Import extraction** (`import`/`multi_import` tags) — given
   source with imports, extraction produces import chunks with
   correct `ImportMeta` metadata.
3. **Mixed extraction** (`mixed` tag) — realistic modules produce
   all expected chunk kinds and method scoping.

The empty-source **property** is checked here for every registered
language; the remaining cross-language invariants (determinism, line
numbers, non-empty content, metadata, error recovery) run over the
committed samples in `test_samples.py`.

Language-specific **edge cases** (java constructors, rust impl,
markdown heading exclusion, etc.) are plain test functions at the
bottom.
"""

from __future__ import annotations

import pytest
from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind, ImportMeta
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import LanguageRegistration, QueryExtraction
from rbtr.languages.treesitter import extract_symbols

# ── Symbol extraction ────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected",
    cases=".cases_extraction",
    has_tag="symbol",
)
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


# ── Mixed extraction ─────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected_kinds, expected_methods",
    cases=".cases_extraction",
    has_tag="mixed",
)
def test_extracts_all_expected_kinds(
    lang: str,
    source: str,
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    kinds = {c.kind for c in chunks}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(c.name, c.scope) for c in chunks if c.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


# ── Import extraction ────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected",
    cases=".cases_extraction",
    has_tag="import",
)
def test_extracts_import_metadata(lang: str, source: str, expected: dict) -> None:
    """First import chunk has the expected metadata."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) >= 1, f"no import chunks extracted from {source!r}"
    assert imports[0].metadata == ImportMeta(**expected)


@parametrize_with_cases(
    "lang, source, count, metadata_list",
    cases=".cases_extraction",
    has_tag="multi_import",
)
def test_extracts_multi_import(
    lang: str,
    source: str,
    count: int,
    metadata_list: list[dict],
) -> None:
    """Multiple imports have correct count and per-import metadata."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == count
    for imp, expected in zip(imports, metadata_list, strict=True):
        assert imp.metadata == ImportMeta(**expected)


# ── Cross-language properties ────────────────────────────────────────


@pytest.mark.parametrize("lang", sorted(get_manager().all_language_ids()))
def test_empty_source_yields_host_presence(lang: str) -> None:
    """Empty source yields one content-less host-presence chunk, for every
    registered language.

    Records the file's host language so the blob-dedup gate skips an empty
    file on later builds instead of re-parsing it every time. The remaining
    cross-language invariants (determinism, line numbers, content, metadata,
    error recovery) run over the committed samples in `test_samples.py`.
    """
    chunks = extract_file(FileEntry("input", "sha1", b""), lang)
    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert chunks[0].language == lang


# ── Language-specific edge cases ─────────────────────────────────────


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


# ── Chunker-specific edge cases ──────────────────────────────────────


def test_toml_dotted_key_scope() -> None:
    """A TOML dotted table splits into last-segment name + preceding scope."""
    src = """\
[tool.ruff.lint]
select = ["E"]
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "toml")
    assert chunks[0].name == "lint"
    assert chunks[0].scope == "tool::ruff"


def test_yaml_no_mapping_fallback() -> None:
    """YAML without mapping falls back to single chunk."""
    chunks = extract_file(FileEntry("input", "sha1", b"- item1\n- item2\n"), "yaml")
    assert len(chunks) == 1


def test_svelte_template_extracted_as_host_chunk() -> None:
    """The SFC markup template is a searchable host (`svelte`) chunk.

    Distinct from the delegated `<script>`/`<style>` chunks: the template is
    the component's own markup, so it carries the host language.
    """
    src = """\
<script lang="ts">
  export let name: string = "world";
</script>

<h1 class="greeting">Hello {name}</h1>
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "svelte")
    host = [c for c in chunks if c.language == "svelte"]
    assert host, "expected a svelte-language chunk for the template markup"
    assert "greeting" in host[0].content


def test_svelte_without_template_still_emits_host_chunk() -> None:
    """A script-only SFC still emits one (empty) host chunk.

    The host chunk records the host language/version for dedup, so a
    svelte-plugin bump invalidates the file even with no template markup.
    """
    src = '<script lang="ts">\n  export const x = 1;\n</script>\n'
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "svelte")
    host = [c for c in chunks if c.language == "svelte"]
    assert len(host) == 1
    assert host[0].content == ""


# ── Markdown link extraction ──────────────────────────────────────────


# ── RST reference extraction ───────────────────────────────────────────
