"""Tests for the Bash language plugin.

Covers function extraction from various shell function syntaxes.
Bash is a base grammar — these tests run unconditionally.
"""

from __future__ import annotations

from tree_sitter import Language

from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.hookspec import LanguageRegistration
from rbtr.plugins.manager import get_manager

# ── Fixtures ─────────────────────────────────────────────────────────

_mgr = get_manager()


def _grammar() -> Language:
    g = _mgr.load_grammar("bash")
    assert g is not None
    return g


def _reg() -> LanguageRegistration:
    reg = _mgr.get_registration("bash")
    assert reg is not None
    return reg


def _extract(source: str, file_path: str = "script.sh") -> list[Chunk]:
    reg = _reg()
    assert reg.query is not None
    return extract_symbols(
        file_path,
        "sha1",
        source.encode(),
        _grammar(),
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
    )


def _symbols(source: str, file_path: str = "script.sh") -> list[tuple[str, str, str]]:
    return [(c.kind, c.name, c.scope) for c in _extract(source, file_path)]


# ── Registration ─────────────────────────────────────────────────────


def test_registration_exists() -> None:
    reg = _reg()
    assert reg.id == "bash"


def test_extensions() -> None:
    reg = _reg()
    assert ".sh" in reg.extensions
    assert ".bash" in reg.extensions
    assert ".zsh" in reg.extensions


def test_filenames() -> None:
    reg = _reg()
    assert "Makefile" in reg.filenames
    assert "Dockerfile" in reg.filenames
    assert ".bashrc" in reg.filenames
    assert ".bash_profile" in reg.filenames
    assert ".zshrc" in reg.filenames


def test_grammar_module() -> None:
    reg = _reg()
    assert reg.grammar_module == "tree_sitter_bash"


def test_no_import_extractor() -> None:
    reg = _reg()
    assert reg.import_extractor is None


def test_empty_scope_types() -> None:
    reg = _reg()
    assert reg.scope_types == frozenset()


# ── Function extraction: keyword syntax ──────────────────────────────


def test_function_keyword_syntax() -> None:
    """function deploy { ... }"""
    src = """\
function deploy {
    echo deploying
}
"""
    syms = _symbols(src)
    assert ("function", "deploy", "") in syms


def test_function_keyword_with_parens() -> None:
    """function deploy() { ... }"""
    src = """\
function deploy() {
    echo deploying
}
"""
    syms = _symbols(src)
    assert ("function", "deploy", "") in syms


# ── Function extraction: POSIX syntax ────────────────────────────────


def test_function_posix_syntax() -> None:
    """deploy() { ... }"""
    src = """\
deploy() {
    echo deploying
}
"""
    syms = _symbols(src)
    assert ("function", "deploy", "") in syms


# ── Multiple functions ───────────────────────────────────────────────


def test_multiple_functions() -> None:
    src = """\
function setup {
    echo setup
}

function teardown {
    echo teardown
}

run() {
    echo run
}
"""
    names = [s[1] for s in _symbols(src) if s[0] == "function"]
    assert "setup" in names
    assert "teardown" in names
    assert "run" in names


# ── No scoping (all top-level) ───────────────────────────────────────


def test_all_functions_are_top_level() -> None:
    """Bash has no classes — all functions should have empty scope."""
    src = """\
function a {
    :;
}
function b {
    :;
}
"""
    scopes = [s[2] for s in _symbols(src)]
    assert all(scope == "" for scope in scopes)


# ── No imports captured ──────────────────────────────────────────────


def test_no_imports_extracted() -> None:
    """source/. commands are not captured as imports."""
    src = """\
source ./env.sh
. /etc/profile
"""
    chunks = _extract(src)
    import_chunks = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert import_chunks == []


# ── Line numbers ─────────────────────────────────────────────────────


def test_line_numbers() -> None:
    src = """\
#!/bin/bash

function deploy {
    echo deploying
}
"""
    chunks = _extract(src)
    fn = next(c for c in chunks if c.name == "deploy")
    assert fn.line_start == 3


# ── Content captured ─────────────────────────────────────────────────


def test_function_content_captured() -> None:
    src = """\
deploy() {
    echo 'going live'
}
"""
    chunks = _extract(src)
    fn = next(c for c in chunks if c.name == "deploy")
    assert "echo 'going live'" in fn.content


# ── Edge cases ───────────────────────────────────────────────────────


def test_empty_source() -> None:
    chunks = _extract("")
    assert chunks == []


def test_script_without_functions() -> None:
    src = """\
#!/bin/bash
echo hello
exit 0
"""
    chunks = _extract(src)
    assert chunks == []


def test_function_with_local_vars() -> None:
    src = """\
setup() {
    local dir="/tmp"
    mkdir -p "$dir"
}
"""
    syms = _symbols(src)
    assert ("function", "setup", "") in syms


def test_function_with_conditionals() -> None:
    src = """\
check() {
    if [ -f /tmp/x ]; then
        echo yes
    fi
}
"""
    syms = _symbols(src)
    assert ("function", "check", "") in syms
