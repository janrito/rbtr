# rbtr-lang-markdown

Markdown support for [rbtr]. A **default** plugin — installed with
rbtr itself (`pip install rbtr`).

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

Two tree-sitter query plugins, one for each half of the grammar:
`markdown` reads the blocks, `markdown_inline` the text inside them.

- **Sections** — headings → doc sections holding their text up to their
  first subsection; `scope` is the `::` chain of enclosing headings. Text
  under no heading becomes fragments.
- **Links** — local `[text](path.md)`, `[text](path.md#fragment)` and
  `[text](#fragment)` → imports (the fragment becomes `names`). A fragment
  reaches the section whose heading it is, as renderers write it
  (`#getting-started` for `## Getting Started`). External links are skipped.
- **Raw HTML** — `<div id="x">` on its own lines and `<a id="x">` in a
  paragraph are HTML, so an element with an `id` or `name` is a section a
  link reaches.
- **Fenced code** — delegated to the block's language (see below).

## Chunks produced

```markdown
# Guide                    <!-- doc_section "Guide"                       -->
## Setup                   <!-- doc_section "Setup", scope "Guide"         -->
See [the API](api.md#run). <!-- import, metadata {module: api.md, names: run} -->
```

## Embedded / injected chunks

A fenced block delegates to its language, extracted at real line numbers —
and delegation **recurses**, so a block that itself embeds another language
resolves all the way down (markdown → html → js):

````markdown
```yaml
service: greeter       <!-- doc_section "service" (yaml)               -->
```
```html
<main>                 <!-- doc_section "main"    (html)               -->
  <script>
    function boot() {} <!-- function "boot"      (javascript, nested)  -->
  </script>
</main>
```
````

## Grammar & dependencies

Uses the `tree-sitter-markdown` grammar. No runtime dependency on other
language plugins; the test suite dev-depends on the plugins its sample's
fenced blocks delegate to (bash, css, html, javascript, python, scss — plus
svelte/toml/yaml via core until those are packaged).
