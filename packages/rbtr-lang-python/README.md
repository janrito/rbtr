# rbtr-lang-python

Python support for [rbtr](../rbtr). A **default** plugin — installed with rbtr
itself (`pip install rbtr`).

## What it ingests

- **Functions & methods** — `def` / `async def`; methods are scoped to their
  enclosing class.
- **Classes** — class definitions (which also form a scope for their members).
- **Variables** — module-level assignments, including flat tuple unpacking.
  Function locals and class attributes stay within their enclosing chunk.
- **Imports** — `import` and `from … import` (relative, aliased, multi-name)
  → import chunks with resolved module + names, for cross-file edges.

A symbol's docstring (the leading `"""…"""`) is folded into its chunk content.

## Chunks produced

```python
def greet(name): ...             # function "greet"
class User:                      # class "User"
    def save(self): ...          #   method "save", scope "User"
MAX = 100                        # variable "MAX"
from .utils import helper        # import, metadata {module: ".utils", names: "helper"}
```

## Embedded / injected chunks

None of its own — Python is embedded *by* Markdown fenced code blocks, which
delegate ` ```python ` blocks here.

## Grammar & dependencies

Uses the `tree-sitter-python` grammar. No dependency on other language plugins.
