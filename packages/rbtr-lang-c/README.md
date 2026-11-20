# rbtr-lang-c

C support for [rbtr]. A **default** plugin — installed with rbtr
itself (`pip install rbtr`).

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

- **Functions** — function definitions and prototypes, including those
  inside a header's include guard or another `#if` / `#ifdef`.
- **Classes** — `struct` / `enum` / `union` and `typedef` types (a
  `typedef struct G G;` yields two: the definition and the alias).
- **Variables** — top-level variables, inside conditionals too, and enum
  constants (C enum constants
  leak into the enclosing scope, so they are file-scope variables).
- **Imports** — `#include <system>` and `#include "local"` headers.
- **Calls** — each function's calls. A call reaching the prototype a header
  declares also reaches the definition in a `.c` or `.cpp` file, which no
  include brings in, and every other function of that name: one defined
  in two programs of a repository reaches both.

## Chunks produced

```c
void greet(const char *name);      /* function "greet"          */
struct Point { int x, y; };        /* class "Point"             */
enum Color { RED, GREEN };         /* class "Color"; variables RED, GREEN */
#include <stdio.h>                 /* import, imports [{module: stdio.h, names: *}]   */
#include "greeter.h"               /* import, imports [{module: greeter.h, names: *}] */
```

## Embedded / injected chunks

None. C does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-c` grammar. No dependency on other language plugins.
