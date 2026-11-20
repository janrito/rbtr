# rbtr-lang-go

Go support for [rbtr]. Optional plugin — install with
`pip install rbtr[go]`.

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

- **Functions & methods** — `func`; a method with a receiver is scoped to its
  receiver type.
- **Classes** — `struct`, `interface`, and named types.
- **Variables** — package-level `var` / `const` (and grouped blocks).
- **Imports** — `import` declarations (single and grouped), for cross-file
  edges. An import path carries the module prefix `go.mod` declares, which is
  read from the repository and stripped, and it names a package — a directory
  — so the edge reaches every `.go` file in it.

Leading `//` doc comments fold into the symbol's content.

## Chunks produced

```go
func Greet(name string) string { … }   // function "Greet"
type Server struct { … }               // class "Server"
func (s *Server) Start() { … }         // method "Start", scope "Server"
const MaxConns = 100                   // variable "MaxConns"
import "fmt"                           // import, imports [{module: fmt}]
```

## Embedded / injected chunks

None. Go does not embed other languages.

## Limitations

rbtr resolves calls by the names they are written with, without types
or control flow. These are what that gets wrong in this language; each
is a strict expected failure, so fixing one fails its test.

| What happens                                                                           | Why                                                                                 | Pinned by                                         |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| A Go import of a directory whose package is named otherwise binds the directory's name | Go binds the declared package name, which is in the imported files, not the import. | `test_an_import_binds_the_packages_declared_name` |

## Grammar & dependencies

Uses the `tree-sitter-go` grammar. No dependency on other language plugins.
