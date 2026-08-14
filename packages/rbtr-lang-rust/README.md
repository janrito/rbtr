# rbtr-lang-rust

Rust support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[rust]`.

## What it ingests

- **Functions & methods** — `fn`; functions inside an `impl` are methods
  scoped to the type.
- **Classes** — `struct`, `enum`, `trait`, and `impl` blocks (an `impl Svc`
  and `struct Svc` both yield a `Svc` class chunk).
- **Variables** — module-level `const` / `static`.
- **Imports** — `use` declarations (scoped paths resolved), for cross-file
  edges.

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
use crate::config::Config;             // import, metadata {module: crate::config::Config}
```

## Embedded / injected chunks

None. Rust does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-rust` grammar. No dependency on other language plugins.
