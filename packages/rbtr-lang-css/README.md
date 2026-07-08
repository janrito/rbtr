# rbtr-lang-css

CSS support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[css]`.

## What it ingests

One chunk per top-level rule set, keyed by its selector. A rule set is a
named collection of property declarations, so it maps to a **class** (the
same shape SQL's `CREATE TABLE` takes) — as do `@media` and `@keyframes`
blocks. `@charset` becomes a config key, `@import` statements become imports
(for cross-language edges), and custom properties become variables.

- **Rule sets & at-rule blocks** — `body { … }`, `.header { … }`, `@media`,
  `@keyframes` → classes (named by selector; `@media` is `<anonymous>`).
- **Directives** — `@charset` → config key.
- **Imports** — `@import url("reset.css")` → an import chunk carrying the
  referenced path, so an edge links to the imported stylesheet.
- **Custom properties** — `--brand: #333;` → variables.

## Chunks produced

`name` is the selector; `scope` is the ancestor rule-set selectors for a
nested rule set (CSS nesting), else empty.

```css
body { color: #333; }              /* class "body",          scope ""      */
.header { background: blue; }      /* class ".header",       scope ""      */
.card { .title { … } }             /* class ".title",        scope ".card" */
@media (max-width: 600px) { … }    /* class "<anonymous>"                 */
@keyframes slide { … }             /* class "slide"                       */
@import url("reset.css");          /* import, metadata {module: reset.css} */
:root { --brand: #333; }           /* variable "--brand"                   */
```

## Embedded / injected chunks

None. CSS does not embed other languages (it is itself embedded by HTML and
the SFC plugins).

## Grammar & dependencies

Uses the `tree-sitter-css` grammar. No dependency on other language plugins;
its `css_nesting_scope` helper is public API that the `scss` and `less`
plugins reuse.
