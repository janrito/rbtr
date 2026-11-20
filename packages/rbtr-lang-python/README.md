# rbtr-lang-python

Python support for [rbtr]. A **default** plugin — installed with rbtr
itself (`pip install rbtr`).

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

- **Functions & methods** — `def` / `async def`; methods are scoped to their
  enclosing class.
- **Classes** — class definitions (which also form a scope for their members).
- **Variables** — module-level assignments; a flat tuple unpacking is one
  variable, named by its names joined. Function locals and class attributes stay within
  their enclosing chunk.
- **Statements** — a module's `if`, `try`, `with`, `for` and `while` blocks,
  and calls spanning lines, each one chunk; a definition or import inside
  one is a chunk nested in it (`if __name__ == "__main__":` holding `def
  main`).
- **Imports** — `import` and `from … import` (relative, aliased, multi-name)
  → import chunks with resolved module + names, for cross-file edges.

A symbol's docstring (the leading `"""…"""`) is folded into its chunk content.

## Chunks produced

```python
def greet(name): ...  # function "greet"


class User:  # class "User"
    def save(self): ...  #   method "save", scope "User"


MAX = 100  # variable "MAX"
low, high = 1, 9  # variable "low, high"
import os, sys  # import, imports [{module: "os"}, {module: "sys"}]
from .utils import helper  # import, imports [{module: ".utils", names: "helper"}]

if __name__ == "__main__":  # statement
    main()
```

## Embedded / injected chunks

None of its own — Python is embedded *by* Markdown fenced code blocks, which
delegate ` ```python ` blocks here.

## Limitations

rbtr resolves calls by the names they are written with, without types
or control flow. These are what that gets wrong in this language; each
is a strict expected failure, so fixing one fails its test.

| What happens                                                                                                      | Why                                            | Pinned by                                    |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| A Python name declared `global` and assigned in a function is read as a local, so a call through it makes no edge | The query reads every assigned name as a local | `case_py_a_name_declared_global_is_no_local` |

## Grammar & dependencies

Uses the `tree-sitter-python` grammar. No dependency on other language plugins.
