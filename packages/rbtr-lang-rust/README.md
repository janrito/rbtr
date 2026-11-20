# rbtr-lang-rust

Rust support for [rbtr]. Optional plugin — install with
`pip install rbtr[rust]`.

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

- **Functions & methods** — `fn`; functions inside an `impl` are methods
  scoped to the type.
- **Classes** — `struct`, `enum`, `trait`, inline `mod x { … }`, and `impl`
  blocks (an `impl Svc` and `struct Svc` both yield a `Svc` class chunk).
- **Variables** — module-level `const` / `static`.
- **Imports** — `use` declarations (scoped paths resolved), and `mod x;`,
  which binds `x` to the module beside the file, for cross-file edges.
- **Attributes** — `#[derive(...)]`, `#[serde(...)]` and the rest fold into
  the item below them, so a type is findable by the traits it derives. A
  module-level `#![allow(...)]`, with no item under it, is its own chunk.

Leading `///` / `//!` doc comments fold into the symbol's content, together
with any attribute between the comment and the item.

## Chunks produced

```rust
fn greet(name: &str) -> String { … }   // function "greet"
struct Server { … }                    // class "Server"
impl Server { fn start(&self) {} }     // class "Server"; method "start", scope "Server"
const MAX: u32 = 100;                  // variable "MAX"
use crate::config::Config;             // import, metadata {module: crate::config, names: Config}
mod util;                              // import, metadata {module: ./util}
```

## Embedded / injected chunks

None. Rust does not embed other languages.

## Limitations

rbtr resolves calls by the names they are written with, without types
or control flow. These are what that gets wrong in this language; each
is a strict expected failure, so fixing one fails its test.

| What happens                                                        | Why                                                         | Pinned by                                           |
| ------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| A Rust call by its full path (`crate::util::log()`) reaches nothing | Its head, `crate`, `self` or `super`, is bound by no `use`. | `test_a_call_by_its_full_path_reaches_the_function` |

## Grammar & dependencies

Uses the `tree-sitter-rust` grammar. No dependency on other language plugins.
