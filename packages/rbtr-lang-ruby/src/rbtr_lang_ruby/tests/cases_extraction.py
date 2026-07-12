"""Ruby extraction test cases."""

from __future__ import annotations

import pytest
from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MultiImportCase = tuple[str, str, int, list[dict[str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_ruby_method_no_args() -> SymbolCase:
    """def greet — top-level function."""
    src = """\
def greet
  puts "hello"
end
"""
    return "ruby", src, [("function", "greet", "")]


@case(tags=["symbol"])
def case_ruby_method_with_args() -> SymbolCase:
    """def add(a, b) — top-level function."""
    src = """\
def add(a, b)
  a + b
end
"""
    return "ruby", src, [("function", "add", "")]


@case(tags=["symbol"])
def case_ruby_multiple_functions() -> SymbolCase:
    """Multiple top-level functions."""
    src = """\
def foo
  1
end

def bar
  2
end
"""
    return "ruby", src, [("function", "foo", ""), ("function", "bar", "")]


@case(tags=["symbol"])
def case_ruby_class() -> SymbolCase:
    """class Shape."""
    src = """\
class Shape
end
"""
    return "ruby", src, [("class", "Shape", "")]


@case(tags=["symbol"])
def case_ruby_module() -> SymbolCase:
    """module Utils."""
    src = """\
module Utils
end
"""
    return "ruby", src, [("class", "Utils", "")]


@case(tags=["symbol"])
def case_ruby_class_superclass() -> SymbolCase:
    """class Circle < Shape."""
    src = """\
class Circle < Shape
end
"""
    return "ruby", src, [("class", "Circle", "")]


@case(tags=["symbol"])
def case_ruby_method_scoped_to_class() -> SymbolCase:
    """Method scoped to class."""
    src = """\
class Foo
  def bar
    1
  end
end
"""
    return "ruby", src, [("method", "bar", "Foo")]


@case(tags=["symbol"])
def case_ruby_singleton_method() -> SymbolCase:
    """def self.build — singleton method."""
    src = """\
class Factory
  def self.build
    new
  end
end
"""
    return "ruby", src, [("method", "build", "Factory")]


@case(tags=["symbol"])
def case_ruby_method_scoped_to_module() -> SymbolCase:
    """Method scoped to module."""
    src = """\
module Helpers
  def format(s)
    s.strip
  end
end
"""
    return "ruby", src, [("method", "format", "Helpers")]


@case(tags=["symbol"])
def case_ruby_top_level_not_scoped() -> SymbolCase:
    """Function after class is not scoped."""
    src = """\
class C
end

def standalone
  1
end
"""
    return "ruby", src, [("function", "standalone", "")]


@case(tags=["symbol"])
def case_ruby_nested_class_in_module() -> SymbolCase:
    """Class nested in module, method scoped to inner class."""
    src = """\
module Utils
  class Parser
    def parse(input)
      input
    end
  end
end
"""
    return (
        "ruby",
        src,
        [
            ("class", "Utils", ""),
            ("class", "Parser", "Utils"),
            ("method", "parse", "Utils::Parser"),
        ],
    )


@case(tags=["symbol"])
def case_ruby_module_in_module() -> SymbolCase:
    """A method in a module nested in a module carries the full path."""
    src = """\
module A
  module B
    def f
      1
    end
  end
end
"""
    return "ruby", src, [("method", "f", "A::B")]


@case(tags=["symbol"])
def case_ruby_module_module_class_method() -> SymbolCase:
    """A method nested module::module::class carries the full path."""
    src = """\
module A
  module B
    class C
      def go
        1
      end
    end
  end
end
"""
    return "ruby", src, [("method", "go", "A::B::C")]


@case(tags=["import"])
def case_ruby_require_simple() -> ImportCase:
    """require "json"."""
    return "ruby", 'require "json"\n', {"module": "json"}


@case(tags=["import"])
def case_ruby_require_nested() -> ImportCase:
    """require "net/http"."""
    return "ruby", 'require "net/http"\n', {"module": "net/http"}


@case(tags=["import"])
def case_ruby_require_relative() -> ImportCase:
    """require_relative "helpers"."""
    return "ruby", 'require_relative "helpers"\n', {"module": "helpers", "dots": "1"}


@case(tags=["import"])
def case_ruby_require_relative_nested() -> ImportCase:
    """require_relative "lib/utils"."""
    return "ruby", 'require_relative "lib/utils"\n', {"module": "lib/utils", "dots": "1"}


@case(tags=["import"])
def case_ruby_require_relative_dot_prefix() -> ImportCase:
    """require_relative "./config" — the `./` is stripped into dots."""
    return "ruby", 'require_relative "./config"\n', {"module": "config", "dots": "1"}


@case(tags=["import"])
def case_ruby_require_relative_parent() -> ImportCase:
    """require_relative "../lib/utils" — `../` becomes dots=2."""
    return "ruby", 'require_relative "../lib/utils"\n', {"module": "lib/utils", "dots": "2"}


@case(tags=["import"])
def case_ruby_require_empty_string() -> ImportCase:
    """require "" — empty string returns empty metadata.

    Covers `ruby.py` lines 53 and 70.
    """
    return "ruby", 'require ""\n', {}


@case(tags=["multi_import"])
def case_ruby_multiple_requires() -> MultiImportCase:
    """require + require_relative."""
    src = """\
require "json"
require_relative "helpers"
"""
    return "ruby", src, 2, [{"module": "json"}, {"module": "helpers", "dots": "1"}]


@case(tags=["mixed"])
def case_ruby_full_file() -> MixedCase:
    """Realistic Ruby file with doc comments on top-level
    declarations.  Comments inside the class body are not
    attached to their methods by the current Ruby grammar (see
    note in `case_docstrings.py`), so only the top-level
    module, class, and `main` carry docs here.  Methods carry the
    full module::class path now that addressing composes the
    enclosing-scope chain.
    """
    src = """\
require "json"
require_relative "config"

# Application namespace for the service.
module App
  # Server runs the request loop.
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

# Entry point used by bin/app.
def main
  server = App::Server.new
  server.start
end
"""
    return (
        "ruby",
        src,
        {"import", "class", "method", "function"},
        [("start", "App::Server"), ("stop", "App::Server"), ("default", "App::Server")],
    )


@case(tags=["symbol"])
def case_ruby_constant() -> SymbolCase:
    """Top-level constant."""
    return "ruby", "MAX_SIZE = 100\n", [("variable", "MAX_SIZE", "")]


@case(tags=["symbol"])
def case_ruby_multiple_assignment() -> SymbolCase:
    """Ruby multiple assignment of constants."""
    return "ruby", "A, B = 1, 2\n", [("variable", "A", ""), ("variable", "B", "")]


@case(tags=["symbol"])
def case_ruby_splat_assignment() -> SymbolCase:
    """Ruby splat target."""
    return "ruby", "A, *B = list\n", [("variable", "A", ""), ("variable", "B", "")]


@case(tags=["symbol"], marks=_xfail_nested)
def case_ruby_nested_unpack_xfail() -> SymbolCase:
    """Ruby nested destructuring — only the outer level captured today."""
    return (
        "ruby",
        "(A, B), C = x\n",
        [("variable", "A", ""), ("variable", "B", ""), ("variable", "C", "")],
    )
