# rbtr-lang-ruby

Ruby support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[ruby]`.

## What it ingests

- **Functions & methods** — `def` (a method inside a class/module is scoped
  to it; top-level `def` is a function).
- **Classes** — `class` and `module` definitions (a module also scopes its
  members).
- **Variables** — constants and module-level assignments.
- **Imports** — `require` and `require_relative`, for cross-file edges.

Leading `#` doc comments fold into the symbol's content.

## Chunks produced

```ruby
def greet(name); …; end          # function "greet"
class User                       # class "User"
  def save; …; end               #   method "save", scope "User"
end
MAX = 100                        # variable "MAX"
require_relative "config"        # import, metadata {module: config}
```

## Embedded / injected chunks

None. Ruby does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-ruby` grammar. No dependency on other language plugins.
