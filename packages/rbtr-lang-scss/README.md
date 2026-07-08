# rbtr-lang-scss

SCSS / Sass support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[scss]`.

## What it ingests

SCSS extends CSS with the Sass preprocessor constructs. Definitions map onto
rbtr's capture conventions by shape:

- **Variables** — `$`-prefixed declarations (`$primary: #333;`).
- **Routines** — `@mixin` and `@function` definitions → functions.
- **Rule sets & at-rule blocks** — selectors (incl. `%placeholder`), `@media`,
  `@keyframes` → classes (a named collection of declarations).
- **Directives** — `@charset` → config key.
- **Imports** — `@use` / `@forward` / `@import` → imports (for cross-file
  edges).

## Chunks produced

`name` is the declaration/selector; `scope` is the ancestor rule-set
selectors for a nested rule (CSS-family nesting), else empty.

```scss
$primary: #333;                /* variable "$primary"                   */
@mixin flex($dir) { … }        /* function "flex"                       */
@function rem($px) { … }       /* function "rem"                        */
%card { … }                    /* class "%card"                         */
.btn { … }                     /* class ".btn"                          */
@keyframes slide { … }         /* class "slide"                         */
@use "config";                 /* import, metadata {module: "config"}   */
```

## Embedded / injected chunks

None. SCSS does not embed other languages (it is embedded by the SFC plugins).

## Grammar & dependencies

Uses the `tree-sitter-scss` grammar. Depends on **rbtr-lang-css** — it reuses
that plugin's public `css_nesting_scope` helper for CSS-family rule nesting.
