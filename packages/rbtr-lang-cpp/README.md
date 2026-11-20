# rbtr-lang-cpp

C++ support for [rbtr]. A **default** plugin — installed with rbtr
itself (`pip install rbtr`).

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

- **Functions** — free functions, function templates and their
  specialisations, and members defined outside their class
  (`Order::~Order()`, `Order::operator<<`).
- **Classes** — classes, structs, and enums.
- **Methods** — member functions, constructors, destructors and operators,
  scoped to their class.
- **Variables** — namespace-level and top-level variables, and with
  prototypes, those inside `#if` / `#ifdef` and `extern "C"` blocks.
- **Imports** — `#include` headers, C's `.h` too.
- **Calls** — each function's calls. A call reaching a function's
  prototype also reaches its definitions, in C++ or C, as in C.

Scope uses `::` and includes the enclosing namespace. CUDA sources (`.cu`,
`.cuh`) are read as C++.

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
