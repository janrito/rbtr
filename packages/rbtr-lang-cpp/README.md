# rbtr-lang-cpp

C++ support for [rbtr](../rbtr). A **default** plugin — installed with rbtr
itself (`pip install rbtr`).

## What it ingests

- **Functions** — free functions and function templates.
- **Classes** — classes, structs, and enums.
- **Methods** — member functions and constructors, scoped to their class.
- **Variables** — namespace-level and top-level variables.
- **Imports** — `#include` headers.

Scope uses `::` and includes the enclosing namespace.

## Chunks produced

```cpp
int add(int a, int b) { … }        // function "add"
namespace ui {                     //
  class Widget { void draw(); };   //   class "Widget"; method "draw" scope "ui::Widget"
  int count = 0;                   //   variable "count", scope "ui"
}
#include <vector>                  // import, metadata {module: vector}
```

## Embedded / injected chunks

None. C++ does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-cpp` grammar. No dependency on other language plugins.
