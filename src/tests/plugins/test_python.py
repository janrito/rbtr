"""Tests for the Python language plugin.

Covers symbol extraction (functions, classes, methods, imports)
and structured import metadata for all Python import forms.
"""

from __future__ import annotations

import pytest
from tree_sitter import Language

from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.hookspec import ImportExtractor, LanguageRegistration
from rbtr.plugins.manager import get_manager

# ── Fixtures ─────────────────────────────────────────────────────────

_mgr = get_manager()


@pytest.fixture
def grammar() -> Language:
    """Tree-sitter Python grammar."""
    g = _mgr.load_grammar("python")
    assert g is not None
    return g


@pytest.fixture
def registration() -> LanguageRegistration:
    """Python LanguageRegistration."""
    reg = _mgr.get_registration("python")
    assert reg is not None
    return reg


@pytest.fixture
def query(registration: LanguageRegistration) -> str:
    """Python query string."""
    assert registration.query is not None
    return registration.query


@pytest.fixture
def extractor(registration: LanguageRegistration) -> ImportExtractor:
    """Python import extractor callable."""
    assert registration.import_extractor is not None
    return registration.import_extractor


@pytest.fixture
def scope_types(registration: LanguageRegistration) -> frozenset[str]:
    """Python scope types."""
    return registration.scope_types


def _extract(
    source: str,
    grammar: Language,
    query: str,
    extractor: ImportExtractor,
    scope_types: frozenset[str],
    file_path: str = "src/app.py",
) -> list[Chunk]:
    return extract_symbols(
        file_path,
        "abc123",
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
    file_path: str = "src/app.py",
) -> list[tuple[str, str, str]]:
    """Return (kind, name, scope) tuples."""
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
    file_path: str = "src/app.py",
) -> list[Chunk]:
    return [
        c
        for c in _extract(source, grammar, query, extractor, scope_types, file_path)
        if c.kind == ChunkKind.IMPORT
    ]


# ── Registration ─────────────────────────────────────────────────────


def test_registration_exists(registration: LanguageRegistration) -> None:
    assert registration.id == "python"


def test_extensions(registration: LanguageRegistration) -> None:
    assert ".py" in registration.extensions
    assert ".pyi" in registration.extensions


def test_grammar_module(registration: LanguageRegistration) -> None:
    assert registration.grammar_module == "tree_sitter_python"


def test_has_query(registration: LanguageRegistration) -> None:
    assert registration.query is not None


def test_has_import_extractor(registration: LanguageRegistration) -> None:
    assert registration.import_extractor is not None


def test_scope_types_contains_class_definition(registration: LanguageRegistration) -> None:
    assert "class_definition" in registration.scope_types


# ── Function extraction ──────────────────────────────────────────────


def test_extract_simple_function(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols(
        """\
def hello():
    pass
""",
        grammar,
        query,
        extractor,
        scope_types,
    )
    assert ("function", "hello", "") in syms


def test_extract_function_with_args(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols(
        """\
def add(a, b):
    return a + b
""",
        grammar,
        query,
        extractor,
        scope_types,
    )
    assert ("function", "add", "") in syms


def test_extract_async_function(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols(
        """\
async def fetch():
    pass
""",
        grammar,
        query,
        extractor,
        scope_types,
    )
    assert ("function", "fetch", "") in syms


def test_extract_multiple_functions(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
def foo():
    pass

def bar():
    pass

def baz():
    pass
"""
    names = [
        s[1] for s in _symbols(src, grammar, query, extractor, scope_types) if s[0] == "function"
    ]
    assert names == ["foo", "bar", "baz"]


def test_extract_decorated_function(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
@decorator
def wrapped():
    pass
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("function", "wrapped", "") in syms


def test_function_line_numbers(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\


def third_line():
    pass
"""
    chunks = _extract(src, grammar, query, extractor, scope_types)
    fn = next(c for c in chunks if c.name == "third_line")
    assert fn.line_start == 3


def test_function_content_captured(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
def greet():
    print('hi')
"""
    chunks = _extract(src, grammar, query, extractor, scope_types)
    fn = next(c for c in chunks if c.name == "greet")
    assert "print('hi')" in fn.content


def test_function_metadata_empty(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    chunks = _extract(
        """\
def f():
    pass
""",
        grammar,
        query,
        extractor,
        scope_types,
    )
    fn = next(c for c in chunks if c.kind == ChunkKind.FUNCTION)
    assert fn.metadata == {}


# ── Class extraction ─────────────────────────────────────────────────


def test_extract_simple_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols(
        """\
class Foo:
    pass
""",
        grammar,
        query,
        extractor,
        scope_types,
    )
    assert ("class", "Foo", "") in syms


def test_extract_class_with_bases(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    syms = _symbols(
        """\
class Bar(Foo, Mixin):
    pass
""",
        grammar,
        query,
        extractor,
        scope_types,
    )
    assert ("class", "Bar", "") in syms


def test_extract_decorated_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
@dataclass
class Config:
    name: str
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("class", "Config", "") in syms


def test_extract_multiple_classes(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class A:
    pass

class B:
    pass
"""
    names = [s[1] for s in _symbols(src, grammar, query, extractor, scope_types) if s[0] == "class"]
    assert names == ["A", "B"]


# ── Method extraction (scoping) ──────────────────────────────────────


def test_method_in_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Foo:
    def bar(self):
        pass
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("method", "bar", "Foo") in syms


def test_multiple_methods_in_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Svc:
    def start(self):
        pass
    def stop(self):
        pass
"""
    methods = [
        (s[1], s[2])
        for s in _symbols(src, grammar, query, extractor, scope_types)
        if s[0] == "method"
    ]
    assert ("start", "Svc") in methods
    assert ("stop", "Svc") in methods


def test_nested_class_method(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Outer:
    class Inner:
        def deep(self):
            pass
"""
    methods = [
        (s[1], s[2])
        for s in _symbols(src, grammar, query, extractor, scope_types)
        if s[0] == "method"
    ]
    assert ("deep", "Inner") in methods


def test_top_level_function_not_method(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Foo:
    pass

def standalone():
    pass
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("function", "standalone", "") in syms


def test_static_method_still_scoped(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class Svc:
    @staticmethod
    def create():
        pass
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("method", "create", "Svc") in syms


# ── Import metadata: bare imports ────────────────────────────────────


def test_import_bare_module(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports("import os\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "os"}


def test_import_dotted_module(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports("import os.path\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "os.path"}


def test_import_deeply_nested(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports("import a.b.c.d\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "a.b.c.d"}


# ── Import metadata: from ... import ────────────────────────────────


def test_from_import_single_name(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports("from pathlib import Path\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "pathlib", "names": "Path"}


def test_from_import_multiple_names(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "from rbtr.index.models import Chunk, Edge\n", grammar, query, extractor, scope_types
    )[0]
    assert imp.metadata["module"] == "rbtr.index.models"
    assert imp.metadata["names"] == "Chunk,Edge"


def test_from_import_aliased(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "from .models import Chunk as C\n", grammar, query, extractor, scope_types, "src/pkg/mod.py"
    )[0]
    assert imp.metadata["names"] == "Chunk"


def test_from_import_multiple_aliased(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "from models import Foo as F, Bar as B\n", grammar, query, extractor, scope_types
    )[0]
    assert imp.metadata["names"] == "Foo,Bar"


# ── Import metadata: relative imports ────────────────────────────────


def test_relative_dot_with_module(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "from .models import Chunk\n",
        grammar,
        query,
        extractor,
        scope_types,
        "src/rbtr/index/store.py",
    )[0]
    assert imp.metadata == {"dots": "1", "module": "models", "names": "Chunk"}


def test_relative_dotdot_with_module(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "from ..core import engine\n",
        grammar,
        query,
        extractor,
        scope_types,
        "src/pkg/sub/mod.py",
    )[0]
    assert imp.metadata == {"dots": "2", "module": "core", "names": "engine"}


def test_relative_dot_only(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "from . import utils\n",
        grammar,
        query,
        extractor,
        scope_types,
        "src/pkg/mod.py",
    )[0]
    assert imp.metadata["dots"] == "1"
    assert "module" not in imp.metadata
    assert imp.metadata["names"] == "utils"


def test_relative_three_dots(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    imp = _imports(
        "from ...lib import helper\n",
        grammar,
        query,
        extractor,
        scope_types,
        "src/a/b/c/mod.py",
    )[0]
    assert imp.metadata["dots"] == "3"
    assert imp.metadata["module"] == "lib"
    assert imp.metadata["names"] == "helper"


# ── Import metadata: edge cases ──────────────────────────────────────


def test_import_star(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    """import * is captured but has no names metadata."""
    imps = _imports("from os.path import *\n", grammar, query, extractor, scope_types)
    assert len(imps) == 1
    assert imps[0].metadata["module"] == "os.path"


def test_multiple_imports(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
import os
import sys
"""
    imps = _imports(src, grammar, query, extractor, scope_types)
    assert len(imps) == 2
    modules = {i.metadata.get("module") for i in imps}
    assert modules == {"os", "sys"}


def test_import_inside_function(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    """Nested imports are still captured."""
    src = """\
def f():
    import json
"""
    imps = _imports(src, grammar, query, extractor, scope_types)
    assert len(imps) == 1
    assert imps[0].metadata == {"module": "json"}


def test_import_inside_class(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
class C:
    from collections import OrderedDict
"""
    imps = _imports(src, grammar, query, extractor, scope_types)
    assert len(imps) == 1


# ── Mixed extraction ─────────────────────────────────────────────────


def test_full_module(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    """A realistic module with all symbol types."""
    src = """\
import os
from pathlib import Path

class Config:
    def __init__(self):
        pass
    def load(self):
        pass

def main():
    pass
"""
    chunks = _extract(src, grammar, query, extractor, scope_types)
    kinds = {c.kind for c in chunks}
    assert ChunkKind.IMPORT in kinds
    assert ChunkKind.CLASS in kinds
    assert ChunkKind.METHOD in kinds
    assert ChunkKind.FUNCTION in kinds

    # Verify scoping.
    init = next(c for c in chunks if c.name == "__init__")
    assert init.scope == "Config"
    main = next(c for c in chunks if c.name == "main")
    assert main.scope == ""


def test_empty_source(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    chunks = _extract("", grammar, query, extractor, scope_types)
    assert chunks == []


def test_syntax_error_partial_parse(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    """Tree-sitter is error-tolerant — valid parts still extract."""
    src = """\
def good():
    pass

def bad(
"""
    chunks = _extract(src, grammar, query, extractor, scope_types)
    names = [c.name for c in chunks if c.kind == ChunkKind.FUNCTION]
    assert "good" in names


def test_blob_sha_propagated(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    chunks = _extract(
        """\
def f():
    pass
""",
        grammar,
        query,
        extractor,
        scope_types,
    )
    assert all(c.blob_sha == "abc123" for c in chunks)


def test_chunk_ids_deterministic(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
def f():
    pass
"""
    c1 = _extract(src, grammar, query, extractor, scope_types)
    c2 = _extract(src, grammar, query, extractor, scope_types)
    assert [c.id for c in c1] == [c.id for c in c2]


def test_chunk_ids_change_with_file_path(
    grammar: Language, query: str, extractor: ImportExtractor, scope_types: frozenset[str]
) -> None:
    src = """\
def f():
    pass
"""
    c1 = _extract(src, grammar, query, extractor, scope_types, file_path="a.py")
    c2 = _extract(src, grammar, query, extractor, scope_types, file_path="b.py")
    assert c1[0].id != c2[0].id
