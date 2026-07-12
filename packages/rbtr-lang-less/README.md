# rbtr-lang-less

Less support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[less]`.

## What it ingests

Less extends CSS with the Less preprocessor constructs. Definitions map onto
rbtr's capture conventions by shape:

- **Variables** — `@`-prefixed declarations (`@primary: #333;`).
- **Routines** — `.mixin()` definitions → functions. Mixin *calls* are
  references, not definitions, and are skipped.
- **Rule sets & at-rule blocks** — selectors, `@media`, `@keyframes` → classes
  (a named collection of declarations).
- **Directives** — `@charset` → config key.
- **Imports** — `@import` → imports (for cross-file edges).

## Chunks produced

`name` is the declaration/selector; `scope` is the ancestor rule-set
selectors for a nested rule (CSS-family nesting), else empty.

```less
@primary: #333;                /* variable "@primary"                 */
.rounded(@r) { … }             /* function "rounded"                  */
.btn { … }                     /* class ".btn"                        */
@keyframes slide { … }         /* class "slide"                       */
@import "reset";               /* import, metadata {module: "reset"}  */
```

## Embedded / injected chunks

None. Less does not embed other languages.

## Grammar & dependencies

Uses the `tree-sitter-less` grammar. Depends on **rbtr-lang-css** — it reuses
that plugin's public `css_nesting_scope` helper for CSS-family rule nesting.
