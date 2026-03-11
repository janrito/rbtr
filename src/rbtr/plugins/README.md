# Language Plugins

rbtr uses a plugin system for language support. Each language gets a
small plugin that teaches the indexer how to detect files, extract
symbols, and understand imports for that language.

## Architecture

```text
src/rbtr/plugins/
  hookspec.py        ← Plugin contract (LanguageRegistration, hookimpl)
  manager.py         ← Loads plugins, builds lookup tables
  python.py          ← Full support: query + import extractor
  test_python.py     ← Co-located tests
  javascript.py      ← JS + TS (shared extractor, separate grammars)
  test_javascript.py
  go.py              ← Full support
  test_go.py
  rust.py            ← Full support
  test_rust.py
  java.py            ← Full support
  test_java.py
  bash.py            ← Functions only (no imports/classes)
  test_bash.py
  defaults.py        ← Detection-only and grammar-only languages
  test_defaults.py
```

Plugins are **not** separate packages — they live in this directory and
share the main `rbtr` package. External plugins are separate packages
that register via entry points (see [External Plugins](#external-plugins)).

## How It Works

A plugin is a class with one method that returns a list of
`LanguageRegistration` objects:

```python
from rbtr.plugins.hookspec import LanguageRegistration, hookimpl

class MyPlugin:
    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [LanguageRegistration(id="mylang", extensions=frozenset({".ml"}))]
```

The `@hookimpl` decorator marks the method for discovery by
[pluggy](https://pluggy.readthedocs.io/). The plugin manager collects
all registrations at startup and builds extension → language lookup
tables.

## Progressive Capability

You don't need to implement everything at once. Each field you add to
`LanguageRegistration` unlocks more analysis:

| What you provide     | What you get                        |
| -------------------- | ----------------------------------- |
| `id` + `extensions`  | File detection, line-based chunks   |
| + `chunker`          | Custom chunking (no grammar needed) |
| + `grammar_module`   | Tree-sitter parse tree              |
| + `query`            | Structural symbol extraction        |
| + `import_extractor` | Precise import edge inference       |
| + `scope_types`      | Method-in-class scoping             |

Start with detection only and add capabilities as needed.

**Two paths to structural extraction:**

- **Tree-sitter** (`grammar_module` + `query`) — the common
  case for programming languages with a grammar package.
- **Custom chunker** (`chunker`) — for prose formats
  (Markdown, RST), languages without a grammar, or
  cases where regex/indentation-based extraction is
  sufficient. The chunker receives `(file_path, blob_sha,

  content)` and returns a list of `Chunk` objects.

## Writing a Built-In Plugin

### Step 1: Find Your Grammar's Node Types

Install the tree-sitter grammar and explore the AST:

```python
import tree_sitter_ruby
from tree_sitter import Language, Parser

lang = Language(tree_sitter_ruby.language())
parser = Parser(lang)

source = b"""
class Greeter
  def hello(name)
    puts "Hello, #{name}"
  end
end
"""

tree = parser.parse(source)

def print_tree(node, indent=0):
    print("  " * indent + f"{node.type} [{node.start_point[0]}:{node.start_point[1]}]")
    for child in node.children:
        print_tree(child, indent + 1)

print_tree(tree.root_node)
```

This shows you the node types you need for your query.

### Step 2: Write the Query

Tree-sitter queries use S-expressions. The indexer recognises these
capture names:

| Capture         | Chunk kind    | Notes                                 |
| --------------- | ------------- | ------------------------------------- |
| `@function`     | `FUNCTION`    | Outer node — captures the full body   |
| `@_fn_name`     | _(name only)_ | Inner node — extracts the symbol name |
| `@class`        | `CLASS`       |                                       |
| `@_cls_name`    | _(name only)_ |                                       |
| `@method`       | `METHOD`      |                                       |
| `@_method_name` | _(name only)_ |                                       |
| `@import`       | `IMPORT`      | Fed to `import_extractor` if provided |

The outer capture (`@function`, `@class`, etc.) defines the chunk
boundaries. The inner name capture (`@_fn_name`, `@_cls_name`, etc.)
extracts the symbol name. Both are required for each symbol type.

Example query for Ruby:

```scheme
(method
  name: (identifier) @_fn_name) @function

(class
  name: (constant) @_cls_name) @class

(call
  method: (identifier) @_import_check
  (#eq? @_import_check "require")) @import
```

### Step 3: Write the Import Extractor (Optional)

An import extractor is a function that takes a tree-sitter `Node`
(the `@import` capture) and returns an `ImportMeta` dict:

```python
from rbtr.index.models import ImportMeta

def extract_import_meta(node: Node) -> ImportMeta:
    """Extract import data from a Ruby require/require_relative."""
    meta: ImportMeta = {}

    # require "json"  →  {"module": "json"}
    # require_relative "./helpers"  →  {"module": "helpers", "dots": "1"}
    for child in node.children:
        if child.type == "argument_list":
            for arg in child.children:
                if arg.type == "string_content" and arg.text:
                    specifier = arg.text.decode()
                    # ... parse specifier into meta ...
                    meta["module"] = specifier
    return meta
```

`ImportMeta` is a `TypedDict` with three optional keys:

| Key      | Type  | Meaning                                          |
| -------- | ----- | ------------------------------------------------ |
| `module` | `str` | Module path — `"os.path"`, `"std/collections"`   |
| `names`  | `str` | Comma-separated symbols — `"Chunk,Edge"`         |
| `dots`   | `str` | Relative depth — `"1"` = current, `"2"` = parent |

The `dots` convention is unified across all languages:

- `1` = current directory (`from .` in Python, `./` in JS)
- `2` = parent directory (`from ..` in Python, `../` in JS, `super::` in Rust)
- `3` = grandparent, etc.

**Shared utilities** for import extractors:

```python
from rbtr.plugins.hookspec import parse_path_relative, collect_scoped_path

# For ./  ../ style paths (JS, TS, CSS):
dots, cleaned = parse_path_relative("../utils/helpers")
# dots=2, cleaned="utils/helpers"

# For :: or . separated scoped paths (Rust, Java):
# Given a scoped_identifier node for std::collections::HashMap:
parts = collect_scoped_path(node)  # ["std", "collections", "HashMap"]
```

If you don't provide an `import_extractor`, the edge inference engine
falls back to text-search: it greps the import statement text for
known chunk names. This works reasonably well and is the default for
languages without a dedicated extractor.

### Step 4: Create the Plugin File

Create `src/rbtr/plugins/ruby.py`:

```python
"""Ruby language plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.plugins.hookspec import LanguageRegistration, hookimpl
from rbtr.index.models import ImportMeta

if TYPE_CHECKING:
    from tree_sitter import Node

_QUERY = """\
(method
  name: (identifier) @_fn_name) @function

(class
  name: (constant) @_cls_name) @class
"""


def extract_import_meta(node: Node) -> ImportMeta:
    meta: ImportMeta = {}
    # ... implementation ...
    return meta


class RubyPlugin:
    """Ruby language support."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="ruby",
                extensions=frozenset({".rb"}),
                grammar_module="tree_sitter_ruby",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class", "module"}),
            ),
        ]
```

### Step 5: Register the Plugin

Add it to `manager.py` in `_register_builtins()`:

```python
from rbtr.plugins.ruby import RubyPlugin

# In _register_builtins():
for plugin_cls in (
    BashPlugin,
    GoPlugin,
    JavaPlugin,
    JavaScriptPlugin,
    PythonPlugin,
    RubyPlugin,  # ← add here
    RustPlugin,
):
    self._pm.register(plugin_cls())
```

If Ruby was previously in `defaults.py`, remove it from there — the
specific plugin overrides it automatically (later registrations win),
but keeping both is unnecessary.

### Step 6: Add the Grammar Dependency

In `pyproject.toml`, add the grammar to `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
languages = [
  # ... existing ...
  "tree-sitter-ruby",
]
```

Base grammars (Python, JSON, YAML, TOML, Bash) are in `[project.dependencies]`
because they're always available. Everything else is optional.

### Step 7: Write Tests

Add tests to `tests/test_index_treesitter.py` (or a new file). Use
`pytest.mark.skipif` for optional grammars:

```python
import pytest

ruby_available = True
try:
    import tree_sitter_ruby  # noqa: F401
except ImportError:
    ruby_available = False

skip_no_ruby = pytest.mark.skipif(not ruby_available, reason="tree-sitter-ruby not installed")

@skip_no_ruby
def test_ruby_extracts_methods():
    from rbtr.index.treesitter import extract_symbols
    # ... test implementation ...

@skip_no_ruby
def test_ruby_import_meta():
    from rbtr.plugins.ruby import extract_import_meta
    # ... test implementation ...
```

### Step 8: Verify

```bash
just check   # runs ruff, mypy, and pytest
```

## External Plugins

External plugins are separate Python packages that register via the
`rbtr.languages` entry-point group. This is for third-party language
support that lives outside the rbtr repository.

### Package Structure

```text
rbtr-lang-ruby/
├── pyproject.toml
└── src/
    └── rbtr_lang_ruby/
        ├── __init__.py
        └── plugin.py
```

### pyproject.toml

```toml
[project]
name = "rbtr-lang-ruby"
version = "0.1.0"
dependencies = [
  "rbtr",
  "tree-sitter-ruby",
]

[project.entry-points."rbtr.languages"]
ruby = "rbtr_lang_ruby.plugin:RubyPlugin"
```

The entry-point key (`ruby`) is arbitrary — it's just a label. The
value points to the plugin class using `module:ClassName` syntax.

### plugin.py

```python
"""Ruby language support for rbtr."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.plugins.hookspec import LanguageRegistration, hookimpl
from rbtr.index.models import ImportMeta

if TYPE_CHECKING:
    from tree_sitter import Node

_QUERY = """\
(method
  name: (identifier) @_fn_name) @function

(class
  name: (constant) @_cls_name) @class
"""


def extract_import_meta(node: Node) -> ImportMeta:
    meta: ImportMeta = {}
    # ... implementation ...
    return meta


class RubyPlugin:
    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="ruby",
                extensions=frozenset({".rb"}),
                grammar_module="tree_sitter_ruby",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class", "module"}),
            ),
        ]
```

### Installation

```bash
# From PyPI (once published):
pip install rbtr-lang-ruby

# From a local checkout:
pip install -e ./rbtr-lang-ruby

# With uv:
uv pip install rbtr-lang-ruby
```

Once installed, rbtr discovers the plugin automatically on next
startup — no configuration needed.

### Precedence

External plugins load **after** all built-in plugins. If an external
plugin registers the same language ID as a built-in, the external one
wins (later registrations override earlier ones). This lets external
plugins replace or enhance built-in support.

Loading order:

1. `DefaultsPlugin` (lowest priority)
2. Built-in specific plugins (Python, JS/TS, Go, Rust, Java, Bash)
3. External plugins via entry points (highest priority)

## Tips

- **Explore the AST first.** The hardest part is finding the right node
  types for your query. Use the tree exploration snippet above, or
  the [tree-sitter playground](https://tree-sitter.github.io/tree-sitter/playground).

- **Start without an import extractor.** Text-search fallback works
  for most languages. Add a structural extractor only when you need
  precise relative import resolution.

- **Check `grammar_entry`.** Most grammar packages expose `language()`,
  but some use a different name. TypeScript uses `language_typescript()`.
  Set `grammar_entry` accordingly.

- **Use `scope_types=frozenset()` for flat languages.** Languages
  without classes (Bash, Lua, etc.) should explicitly set empty scope
  types so the indexer doesn't try to detect method scoping.

- **Look at existing plugins.** The built-in plugins in this directory
  cover the common patterns: `python.py` for a classic OOP language,
  `javascript.py` for two languages sharing an extractor,
  `go.py` for a language with grouped imports, `rust.py` for
  `::` scoped paths, and `bash.py` for a minimal functions-only plugin.
