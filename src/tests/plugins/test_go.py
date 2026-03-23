"""Tests for the Go language plugin.

Covers functions, methods, type declarations, and all import forms.
Skipped when tree-sitter-go is not installed.
"""

from __future__ import annotations

import pytest
from tree_sitter import Language

from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.hookspec import ImportExtractor, LanguageRegistration
from rbtr.plugins.manager import get_manager

# ── Grammar availability ─────────────────────────────────────────────

_mgr = get_manager()
_has_go = _mgr.load_grammar("go") is not None
skip_no_go = pytest.mark.skipif(not _has_go, reason="tree-sitter-go not installed")
pytestmark = skip_no_go

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def grammar() -> Language:
    g = _mgr.load_grammar("go")
    assert g is not None
    return g


@pytest.fixture
def registration() -> LanguageRegistration:
    reg = _mgr.get_registration("go")
    assert reg is not None
    return reg


@pytest.fixture
def query(registration: LanguageRegistration) -> str:
    assert registration.query is not None
    return registration.query


@pytest.fixture
def extractor(registration: LanguageRegistration) -> ImportExtractor:
    assert registration.import_extractor is not None
    return registration.import_extractor


@pytest.fixture
def scope_types(registration: LanguageRegistration) -> frozenset[str]:
    return registration.scope_types


# ── Helpers ──────────────────────────────────────────────────────────


def _extract(
    source: str,
    grammar: Language,
    query: str,
    extractor: ImportExtractor,
    scope_types: frozenset[str],
    file_path: str = "main.go",
) -> list[Chunk]:
    return extract_symbols(
        file_path,
        "sha1",
        source.encode(),
        grammar,
        query,
        import_extractor=extractor,
        scope_types=scope_types,
    )


def _symbols(
    source: str,
    grammar: Language,
    query: str,
    extractor: ImportExtractor,
    scope_types: frozenset[str],
    file_path: str = "main.go",
) -> list[tuple[str, str, str]]:
    return [
        (c.kind, c.name, c.scope)
        for c in _extract(source, grammar, query, extractor, scope_types, file_path)
    ]


def _imports(
    source: str,
    grammar: Language,
    query: str,
    extractor: ImportExtractor,
    scope_types: frozenset[str],
    file_path: str = "main.go",
) -> list[Chunk]:
    return [
        c
        for c in _extract(source, grammar, query, extractor, scope_types, file_path)
        if c.kind == ChunkKind.IMPORT
    ]


# ── Registration ─────────────────────────────────────────────────────


def test_registration_exists(registration: LanguageRegistration) -> None:
    assert registration.id == "go"


def test_extensions(registration: LanguageRegistration) -> None:
    assert ".go" in registration.extensions


def test_grammar_module(registration: LanguageRegistration) -> None:
    assert registration.grammar_module == "tree_sitter_go"


def test_scope_types_contain_type_spec(registration: LanguageRegistration) -> None:
    assert "type_spec" in registration.scope_types


# ── Function extraction ──────────────────────────────────────────────


def test_extract_function(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

func hello() {}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("function", "hello", "") in syms


def test_extract_function_with_params(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

func add(a int, b int) int { return a + b }
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("function", "add", "") in syms


def test_extract_multiple_functions(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

func foo() {}
func bar() {}
func baz() {}
"""
    names = [
        s[1] for s in _symbols(src, grammar, query, extractor, scope_types) if s[0] == "function"
    ]
    assert names == ["foo", "bar", "baz"]


# ── Method extraction ────────────────────────────────────────────────


def test_extract_method(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

type User struct{}

func (u User) Name() string { return u.name }
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("method", "Name", "") in syms


def test_extract_pointer_receiver_method(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

type Svc struct{}

func (s *Svc) Start() {}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("method", "Start", "") in syms


# ── Type declaration extraction ──────────────────────────────────────


def test_extract_struct(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

type User struct {
    Name string
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("class", "User", "") in syms


def test_extract_interface(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

type Reader interface {
    Read(p []byte) (int, error)
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("class", "Reader", "") in syms


def test_extract_type_alias(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

type ID string
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("class", "ID", "") in syms


def test_extract_multiple_types(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

type Foo struct{}
type Bar struct{}
"""
    names = [s[1] for s in _symbols(src, grammar, query, extractor, scope_types) if s[0] == "class"]
    assert "Foo" in names
    assert "Bar" in names


# ── Import: single ───────────────────────────────────────────────────


def test_import_single(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main
import "fmt"
"""
    imp = _imports(src, grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "fmt"}


def test_import_single_nested(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main
import "os/exec"
"""
    imp = _imports(src, grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "os/exec"}


def test_import_single_url(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main
import "github.com/user/repo"
"""
    imp = _imports(src, grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "github.com/user/repo"}


# ── Import: grouped ──────────────────────────────────────────────────


def test_import_grouped(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main
import (
    "fmt"
    "os"
)
"""
    imp = _imports(src, grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "fmt,os"}


def test_import_grouped_with_paths(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main
import (
    "fmt"
    "os/exec"
    "net/http"
)
"""
    imp = _imports(src, grammar, query, extractor, scope_types)[0]
    modules = imp.metadata["module"].split(",")
    assert "fmt" in modules
    assert "os/exec" in modules
    assert "net/http" in modules


def test_import_grouped_single_item(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main
import (
    "fmt"
)
"""
    imp = _imports(src, grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "fmt"}


# ── Import: aliased ──────────────────────────────────────────────────


def test_import_aliased(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main
import f "fmt"
"""
    imp = _imports(src, grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "fmt"}


# ── Mixed extraction ─────────────────────────────────────────────────


def test_full_module(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
package main

import (
    "fmt"
    "os"
)

type Config struct {
    Name string
}

func (c Config) String() string { return c.Name }

func main() {
    fmt.Println("hello")
}
"""
    chunks = _extract(src, grammar, query, extractor, scope_types)
    kinds = {c.kind for c in chunks}
    assert ChunkKind.IMPORT in kinds
    assert ChunkKind.CLASS in kinds
    assert ChunkKind.METHOD in kinds
    assert ChunkKind.FUNCTION in kinds


def test_empty_source(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    chunks = _extract("", grammar, query, extractor, scope_types)
    assert chunks == []
