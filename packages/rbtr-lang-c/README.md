# rbtr-lang-c

C support for [rbtr](../rbtr). A **default** plugin — installed with rbtr
itself (`pip install rbtr`).

## What it ingests

- **Functions** — function definitions and prototypes.
- **Classes** — `struct` / `enum` / `union` and `typedef` types (a
  `typedef struct G G;` yields two: the definition and the alias).
- **Variables** — top-level variables and enum constants (C enum constants
  leak into the enclosing scope, so they are file-scope variables).
- **Imports** — `#include <system>` and `#include "local"` headers.

## Chunks produced

```c
void greet(const char *name);      /* function "greet"          */
struct Point { int x, y; };        /* class "Point"             */
enum Color { RED, GREEN };         /* class "Color"; variables RED, GREEN */
#include <stdio.h>                 /* import, metadata {module: stdio.h}   */
#include "greeter.h"               /* import, metadata {module: greeter.h} */
```

## Embedded / injected chunks

None. C does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-c` grammar. No dependency on other language plugins.
