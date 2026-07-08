# rbtr-lang-go

Go support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[go]`.

## What it ingests

- **Functions & methods** — `func`; a method with a receiver is scoped to its
  receiver type.
- **Classes** — `struct`, `interface`, and named types.
- **Variables** — package-level `var` / `const` (and grouped blocks).
- **Imports** — `import` declarations (single and grouped), for cross-file
  edges.

Leading `//` doc comments fold into the symbol's content.

## Chunks produced

```go
func Greet(name string) string { … }   // function "Greet"
type Server struct { … }               // class "Server"
func (s *Server) Start() { … }         // method "Start", scope "Server"
const MaxConns = 100                   // variable "MaxConns"
import "fmt"                           // import, metadata {module: fmt}
```

## Embedded / injected chunks

None. Go does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-go` grammar. No dependency on other language plugins.
