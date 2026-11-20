# rbtr-lang-ruby

Ruby support for [rbtr]. Optional plugin — install with
`pip install rbtr[ruby]`.

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

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
require_relative "config"        # import, metadata {module: ./config, names: *}
```

## Embedded / injected chunks

None. Ruby does not embed other languages.

## Limitations

rbtr resolves calls by the names they are written with, without types
or control flow. These are what that gets wrong in this language; each
is a strict expected failure, so fixing one fails its test.

| What happens                                                                     | Why                                                                                                        | Pinned by                                                   |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| A Ruby name used before the assignment that makes it a local is read as a local  | A function's locals hold for its whole body, whatever the order                                            | `case_rb_a_name_before_its_assignment_is_a_call`            |
| A call to a Ruby class also reaches a spec block named after it                  | `RSpec.describe Greeter` is a class chunk named `Greeter`, and a call reaches every definition of its name | `test_a_constructor_call_reaches_the_class_not_its_spec`    |
| Of modules one Ruby `include` lists (`include A, B`), the last is searched first | Ruby searches bases last written first, but one `include` puts its first module first                      | `test_one_include_listing_two_modules_puts_the_first_first` |

## Grammar & dependencies

Uses the `tree-sitter-ruby` grammar. No dependency on other language plugins.
