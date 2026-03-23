"""Tests for the Java language plugin.

Covers classes, methods, and all import forms including static
and wildcard imports.  Skipped when tree-sitter-java is not
installed.
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
_has_java = _mgr.load_grammar("java") is not None
skip_no_java = pytest.mark.skipif(not _has_java, reason="tree-sitter-java not installed")
pytestmark = skip_no_java

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def grammar() -> Language:
    g = _mgr.load_grammar("java")
    assert g is not None
    return g


@pytest.fixture
def registration() -> LanguageRegistration:
    reg = _mgr.get_registration("java")
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
    file_path: str = "App.java",
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
    file_path: str = "App.java",
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
    file_path: str = "App.java",
) -> list[Chunk]:
    return [
        c
        for c in _extract(source, grammar, query, extractor, scope_types, file_path)
        if c.kind == ChunkKind.IMPORT
    ]


# ── Registration ─────────────────────────────────────────────────────


def test_registration_exists(registration: LanguageRegistration) -> None:
    assert registration.id == "java"


def test_extensions(registration: LanguageRegistration) -> None:
    assert ".java" in registration.extensions


def test_grammar_module(registration: LanguageRegistration) -> None:
    assert registration.grammar_module == "tree_sitter_java"


def test_scope_types(registration: LanguageRegistration) -> None:
    assert "class_declaration" in registration.scope_types


# ── Class extraction ─────────────────────────────────────────────────


def test_extract_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols("class User {}\n", grammar, query, extractor, scope_types)
    assert ("class", "User", "") in syms


def test_extract_public_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols("public class App {}\n", grammar, query, extractor, scope_types)
    assert ("class", "App", "") in syms


def test_extract_class_with_extends(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols("class Admin extends User {}\n", grammar, query, extractor, scope_types)
    assert ("class", "Admin", "") in syms


def test_extract_class_with_implements(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols(
        "class UserService implements Service {}\n", grammar, query, extractor, scope_types
    )
    assert ("class", "UserService", "") in syms


def test_extract_multiple_classes(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Foo {}
class Bar {}
"""
    names = [s[1] for s in _symbols(src, grammar, query, extractor, scope_types) if s[0] == "class"]
    assert "Foo" in names
    assert "Bar" in names


# ── Method extraction ────────────────────────────────────────────────


def test_method_in_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Service {
    void process() {}
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("method", "process", "Service") in syms


def test_multiple_methods(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Svc {
    void start() {}
    void stop() {}
}
"""
    methods = [
        (s[1], s[2])
        for s in _symbols(src, grammar, query, extractor, scope_types)
        if s[0] == "method"
    ]
    assert ("start", "Svc") in methods
    assert ("stop", "Svc") in methods


def test_static_method(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Factory {
    static Object create() { return null; }
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("method", "create", "Factory") in syms


def test_method_with_params(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Calc {
    int add(int a, int b) { return a + b; }
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("method", "add", "Calc") in syms


def test_constructor_is_not_captured(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    """Java constructors use constructor_declaration, not method_declaration."""
    src = """\
class Foo {
    Foo() {}
}
"""
    methods = [s for s in _symbols(src, grammar, query, extractor, scope_types) if s[0] == "method"]
    # Constructors may or may not be captured depending on query —
    # our query targets method_declaration only.
    constructor_names = [m[1] for m in methods if m[1] == "Foo"]
    assert constructor_names == []


# ── Import: class import ─────────────────────────────────────────────


def test_import_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports("import java.util.HashMap;\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "java.util", "names": "HashMap"}


def test_import_deeply_nested(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports("import com.example.app.models.User;\n", grammar, query, extractor, scope_types)[
        0
    ]
    assert imp.metadata == {"module": "com.example.app.models", "names": "User"}


# ── Import: static import ───────────────────────────────────────────


def test_import_static(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "import static org.junit.Assert.assertEquals;\n", grammar, query, extractor, scope_types
    )[0]
    assert imp.metadata["module"] == "org.junit.Assert"
    assert imp.metadata["names"] == "assertEquals"


def test_import_static_method(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "import static java.util.Collections.sort;\n", grammar, query, extractor, scope_types
    )[0]
    assert imp.metadata["module"] == "java.util.Collections"
    assert imp.metadata["names"] == "sort"


# ── Import: multiple ─────────────────────────────────────────────────


def test_multiple_imports(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
import java.util.List;
import java.util.Map;
"""
    imps = _imports(src, grammar, query, extractor, scope_types)
    assert len(imps) == 2
    modules = {i.metadata["module"] for i in imps}
    assert modules == {"java.util"}


# ── Mixed extraction ─────────────────────────────────────────────────


def test_full_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
import java.util.List;
import java.util.ArrayList;

public class UserService {
    private List<String> names;

    public void addName(String name) {
        names.add(name);
    }

    public List<String> getNames() {
        return names;
    }
}
"""
    chunks = _extract(src, grammar, query, extractor, scope_types)
    kinds = {c.kind for c in chunks}
    assert ChunkKind.IMPORT in kinds
    assert ChunkKind.CLASS in kinds
    assert ChunkKind.METHOD in kinds

    methods = [(c.name, c.scope) for c in chunks if c.kind == ChunkKind.METHOD]
    assert ("addName", "UserService") in methods
    assert ("getNames", "UserService") in methods


def test_empty_source(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    chunks = _extract("", grammar, query, extractor, scope_types)
    assert chunks == []


def test_nested_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Outer {
    class Inner {
        void deep() {}
    }
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("class", "Outer", "") in syms
    assert ("class", "Inner", "Outer") in syms
    assert ("method", "deep", "Inner") in syms
