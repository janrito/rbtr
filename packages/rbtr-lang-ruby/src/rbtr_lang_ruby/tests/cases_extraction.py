"""Ruby extraction test cases."""

from __future__ import annotations

import pytest
from pytest_cases import case

type SymbolCase = tuple[dict[str, str], list[tuple[str, str, str]]]
type ImportCase = tuple[dict[str, str], dict[str, str | list[dict[str, str]]]]
type MultiImportCase = tuple[dict[str, str], int, list[dict[str, str]]]
type MixedCase = tuple[dict[str, str], set[str], list[tuple[str, str]]]

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
    return {"input.rb": src}, [("function", "greet", "")]


@case(tags=["symbol"])
def case_ruby_method_with_args() -> SymbolCase:
    """def add(a, b) — top-level function."""
    src = """\
def add(a, b)
  a + b
end
"""
    return {"input.rb": src}, [("function", "add", "")]


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
    return {"input.rb": src}, [("function", "foo", ""), ("function", "bar", "")]


@case(tags=["symbol"])
def case_ruby_class() -> SymbolCase:
    """class Shape."""
    src = """\
class Shape
end
"""
    return {"input.rb": src}, [("class", "Shape", "")]


@case(tags=["symbol"])
def case_ruby_module() -> SymbolCase:
    """module Utils."""
    src = """\
module Utils
end
"""
    return {"input.rb": src}, [("class", "Utils", "")]


@case(tags=["symbol"])
def case_ruby_class_superclass() -> SymbolCase:
    """class Circle < Shape."""
    src = """\
class Circle < Shape
end
"""
    return {"input.rb": src}, [("class", "Circle", "")]


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
    return {"input.rb": src}, [("method", "bar", "Foo")]


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
    return {"input.rb": src}, [("method", "build", "Factory")]


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
    return {"input.rb": src}, [("method", "format", "Helpers")]


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
    return {"input.rb": src}, [("function", "standalone", "")]


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
        {"input.rb": src},
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
    return {"input.rb": src}, [("method", "f", "A::B")]


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
    return {"input.rb": src}, [("method", "go", "A::B::C")]


@case(tags=["import"])
def case_ruby_require_simple() -> ImportCase:
    """require "json"."""
    return {"input.rb": 'require "json"\n'}, {"module": "json", "names": "*"}


@case(tags=["import"])
def case_ruby_require_nested() -> ImportCase:
    """require "net/http"."""
    return {"input.rb": 'require "net/http"\n'}, {"module": "net/http", "names": "*"}


@case(tags=["import"])
def case_ruby_require_relative() -> ImportCase:
    """require_relative "helpers"."""
    return {"input.rb": 'require_relative "helpers"\n'}, {"module": "./helpers", "names": "*"}


@case(tags=["import"])
def case_ruby_require_relative_nested() -> ImportCase:
    """require_relative "lib/utils"."""
    return {"input.rb": 'require_relative "lib/utils"\n'}, {"module": "./lib/utils", "names": "*"}


@case(tags=["import"])
def case_ruby_require_relative_dot_prefix() -> ImportCase:
    """require_relative "./config" — already written relative."""
    return {"input.rb": 'require_relative "./config"\n'}, {"module": "./config", "names": "*"}


@case(tags=["import"])
def case_ruby_require_relative_parent() -> ImportCase:
    """require_relative "../lib/utils" — already written relative."""
    return {"input.rb": 'require_relative "../lib/utils"\n'}, {
        "module": "../lib/utils",
        "names": "*",
    }


@case(tags=["multi_import"])
def case_ruby_multiple_requires() -> MultiImportCase:
    """require + require_relative."""
    src = """\
require "json"
require_relative "helpers"
"""
    return (
        {"input.rb": src},
        2,
        [{"module": "json", "names": "*"}, {"module": "./helpers", "names": "*"}],
    )


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
        {"input.rb": src},
        {"import", "class", "method", "function"},
        [("start", "App::Server"), ("stop", "App::Server"), ("default", "App::Server")],
    )


@case(tags=["symbol"])
def case_ruby_constant() -> SymbolCase:
    """Top-level constant."""
    return {"input.rb": "MAX_SIZE = 100\n"}, [("variable", "MAX_SIZE", "")]


@case(tags=["symbol"])
def case_ruby_multiple_assignment() -> SymbolCase:
    """Ruby multiple assignment of constants."""
    return {"input.rb": "A, B = 1, 2\n"}, [("variable", "A", ""), ("variable", "B", "")]


@case(tags=["symbol"])
def case_ruby_splat_assignment() -> SymbolCase:
    """Ruby splat target."""
    return {"input.rb": "A, *B = list\n"}, [("variable", "A", ""), ("variable", "B", "")]


@case(tags=["symbol"], marks=_xfail_nested)
def case_ruby_nested_unpack_xfail() -> SymbolCase:
    """Ruby nested destructuring — only the outer level captured today."""
    return (
        {"input.rb": "(A, B), C = x\n"},
        [("variable", "A", ""), ("variable", "B", ""), ("variable", "C", "")],
    )


type ReferencesCase = tuple[str, str, list[tuple[str, str, list[str], list[str]]]]


@case(tags=["references"])
def case_ruby_calls_and_bases() -> ReferencesCase:
    """Each call shape, `Klass.new` as the class it constructs, and a base by path.

    `items.first.size` has no parentheses, so its text is names only and it
    is a path; its head is a local, so it reaches nothing.
    """
    src = """\
class Order < Shop::Base
  def save
    self.check
    check()
    helper 1
    Foo.new(1)
    Shop::Klass.new
    Util.log
    items.first.size
  end
end
"""
    return (
        "ruby",
        src,
        [
            ("class", "Order", [], ["unqualified:Shop.Base"]),
            (
                "method",
                "save",
                [
                    "member:check",
                    "unqualified:check",
                    "unqualified:helper",
                    "unqualified:Foo",
                    "unqualified:Shop.Klass",
                    "unqualified:Util.log",
                    "unqualified:items.first",
                    "unqualified:items.first.size",
                ],
                [],
            ),
        ],
    )


@case(tags=["references"])
def case_rb_a_bare_name_is_a_call_unless_it_is_a_local() -> ReferencesCase:
    """`check` alone calls a method; a parameter, block parameter or assigned name is a local, a value."""
    src = """\
class Order
  def save(a, key: 1, &blk)
    check
    x = compute
    log(x, other)
    return result if pending
    items.each { |item| render(item) }
  end
end
"""
    return (
        "ruby",
        src,
        [
            (
                "method",
                "save",
                [
                    "unqualified:check",
                    "unqualified:compute",
                    "unqualified:log",
                    "type:x",
                    "unqualified:other",
                    "unqualified:result",
                    "unqualified:pending",
                    "unqualified:items.each",
                    "unqualified:render",
                    "type:item",
                ],
                [],
            )
        ],
    )


@case(
    tags=["references"],
    marks=pytest.mark.xfail(
        strict=True,
        reason="a function's locals hold for its whole body, but Ruby makes a name "
        "a local only from the assignment on",
    ),
)
def case_rb_a_name_before_its_assignment_is_a_call() -> ReferencesCase:
    """`check` then `check = 1`: the first `check` is still a method call."""
    src = """\
def save
  check
  check = 1
end
"""
    return "ruby", src, [("function", "save", ["unqualified:check"], [])]


@case(tags=["symbol"])
def case_rb_rspec_groups_and_examples_are_tests() -> SymbolCase:
    """`describe Greeter`, `context "..."` and `it "..."` are tests, named by their description."""
    src = """\
RSpec.describe Greeter do
  context "with a name" do
    it "greets" do
      expect(Greeter.new.greet).to eq("hi")
    end
  end
end
"""
    return (
        {"input.rb": src},
        [
            ("test_function", "Greeter", ""),
            ("test_function", "with a name", ""),
            ("test_function", "greets", ""),
        ],
    )
