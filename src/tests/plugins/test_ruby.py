"""Tests for the Ruby language plugin."""

from __future__ import annotations

import pytest
from tree_sitter import Language, Parser

from rbtr.index.models import Chunk, ChunkKind, ImportMeta
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.ruby import RubyPlugin, extract_import_meta

# ── Helpers ──────────────────────────────────────────────────────────

_HAS_GRAMMAR = True
try:
    import tree_sitter_ruby

    _LANG = Language(tree_sitter_ruby.language())
except ImportError:
    _HAS_GRAMMAR = False
    _LANG = None  # type: ignore[assignment]

needs_grammar = pytest.mark.skipif(not _HAS_GRAMMAR, reason="tree-sitter-ruby not installed")


def _extract(code: str) -> list[Chunk]:
    reg = RubyPlugin().rbtr_register_languages()[0]
    assert reg.query is not None
    return extract_symbols(
        "test.rb",
        "abc123",
        code.encode(),
        _LANG,
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
    )


def _parse_require(code: str) -> ImportMeta:
    """Parse code and extract import metadata from the first require."""
    parser = Parser(_LANG)
    tree = parser.parse(code.encode())
    for child in tree.root_node.children:
        if child.type == "call" and child.text:
            text = child.text.decode()
            if text.startswith("require"):
                return extract_import_meta(child)
    return {}


# ── Registration ─────────────────────────────────────────────────────


def test_registration() -> None:
    regs = RubyPlugin().rbtr_register_languages()
    assert len(regs) == 1
    reg = regs[0]
    assert reg.id == "ruby"
    assert reg.extensions == frozenset({".rb"})
    assert reg.grammar_module == "tree_sitter_ruby"
    assert reg.query is not None
    assert reg.import_extractor is not None
    assert "class" in reg.scope_types
    assert "module" in reg.scope_types


# ── Require extraction ───────────────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('require "json"', {"module": "json"}),
        ('require "net/http"', {"module": "net/http"}),
        ('require_relative "helpers"', {"module": "helpers", "dots": "1"}),
        ('require_relative "lib/utils"', {"module": "lib/utils", "dots": "1"}),
    ],
    ids=["simple", "nested-path", "relative", "relative-nested"],
)
def test_require_meta(code: str, expected: ImportMeta) -> None:
    assert _parse_require(code) == expected


@needs_grammar
def test_multiple_requires() -> None:
    chunks = _extract("""\
require "json"
require_relative "helpers"
""")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 2
    assert imports[0].metadata == {"module": "json"}
    assert imports[1].metadata == {"module": "helpers", "dots": "1"}


# ── Method extraction ────────────────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected_name"),
    [
        (
            """\
def greet
  puts "hello"
end
""",
            "greet",
        ),
        (
            """\
def add(a, b)
  a + b
end
""",
            "add",
        ),
    ],
    ids=["no-args", "with-args"],
)
def test_top_level_function(code: str, expected_name: str) -> None:
    fns = [c for c in _extract(code) if c.kind == ChunkKind.FUNCTION]
    assert len(fns) == 1
    assert fns[0].name == expected_name
    assert fns[0].scope == ""


@needs_grammar
def test_multiple_functions() -> None:
    chunks = _extract("""\
def foo
  1
end

def bar
  2
end
""")
    names = {c.name for c in chunks if c.kind == ChunkKind.FUNCTION}
    assert names == {"foo", "bar"}


# ── Class and module extraction ──────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected_name"),
    [
        (
            """\
class Shape
end
""",
            "Shape",
        ),
        (
            """\
module Utils
end
""",
            "Utils",
        ),
        (
            """\
class Circle < Shape
end
""",
            "Circle",
        ),
    ],
    ids=["class", "module", "class-with-superclass"],
)
def test_class_like(code: str, expected_name: str) -> None:
    classes = [c for c in _extract(code) if c.kind == ChunkKind.CLASS]
    assert any(c.name == expected_name for c in classes)


# ── Method scoping ───────────────────────────────────────────────────


@needs_grammar
def test_method_scoped_to_class() -> None:
    chunks = _extract("""\
class Foo
  def bar
    1
  end
end
""")
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD]
    assert len(methods) == 1
    assert methods[0].name == "bar"
    assert methods[0].scope == "Foo"


@needs_grammar
def test_singleton_method_scoped() -> None:
    chunks = _extract("""\
class Factory
  def self.build
    new
  end
end
""")
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD]
    assert len(methods) == 1
    assert methods[0].name == "build"
    assert methods[0].scope == "Factory"


@needs_grammar
def test_method_scoped_to_module() -> None:
    chunks = _extract("""\
module Helpers
  def format(s)
    s.strip
  end
end
""")
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD]
    assert len(methods) == 1
    assert methods[0].name == "format"
    assert methods[0].scope == "Helpers"


@needs_grammar
def test_nested_class_in_module() -> None:
    chunks = _extract("""\
module Utils
  class Parser
    def parse(input)
      input
    end
  end
end
""")
    classes = [c for c in chunks if c.kind == ChunkKind.CLASS]
    class_names = {c.name for c in classes}
    assert "Utils" in class_names
    assert "Parser" in class_names

    # Parser is scoped to Utils.
    parser_cls = next(c for c in classes if c.name == "Parser")
    assert parser_cls.scope == "Utils"

    # parse() is scoped to Parser.
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD]
    assert len(methods) == 1
    assert methods[0].name == "parse"
    assert methods[0].scope == "Parser"


@needs_grammar
def test_top_level_function_not_scoped() -> None:
    """Functions outside classes/modules should not be methods."""
    chunks = _extract("""\
class C
end

def standalone
  1
end
""")
    fns = [c for c in chunks if c.kind == ChunkKind.FUNCTION]
    assert len(fns) == 1
    assert fns[0].name == "standalone"
    assert fns[0].scope == ""


# ── Full file ────────────────────────────────────────────────────────


@needs_grammar
def test_full_ruby_file() -> None:
    chunks = _extract("""\
require "json"
require_relative "config"

module App
  class Server
    def start
      puts "starting"
    end

    def stop
      puts "stopping"
    end

    def self.default
      new
    end
  end
end

def main
  server = App::Server.new
  server.start
end
""")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    classes = [c for c in chunks if c.kind == ChunkKind.CLASS]
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD]
    fns = [c for c in chunks if c.kind == ChunkKind.FUNCTION]

    assert len(imports) == 2
    assert {c.metadata.get("module") for c in imports} == {"json", "config"}

    assert {c.name for c in classes} == {"App", "Server"}
    server_cls = next(c for c in classes if c.name == "Server")
    assert server_cls.scope == "App"

    assert {m.name for m in methods} == {"start", "stop", "default"}
    assert all(m.scope == "Server" for m in methods)

    assert len(fns) == 1
    assert fns[0].name == "main"
    assert fns[0].scope == ""
