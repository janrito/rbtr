# rbtr-lang-markdown

Markdown support for [rbtr]. A **default** plugin — installed with
rbtr itself (`pip install rbtr`).

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

Two tree-sitter query plugins, one for each half of the grammar:
`markdown` reads the blocks, `markdown_inline` the text inside them.

- **Sections** — headings, `#` or underlined with `=` or `-` → doc
  sections holding their text up to the next heading; `scope` is the `::`
  chain of the headings above of a higher level (`===` is level 1, `---`
  level 2). A heading inside a list item or block quote is a section within
  the one around it. Text under no heading becomes fragments.
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
See [the API](api.md#run). <!-- import, imports [{module: api.md, names: run}] -->
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

## Limitations

Each is a strict expected failure, so fixing one fails its test.

| What happens                                                                             | Why                                                                                                              | Pinned by                                                       |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| A heading inside a list item or block quote has an empty scope, not the heading above it | Its scope is read from the grammar's sections around it, and a top-level heading is read flat, naming no section | `test_md_a_nested_heading_is_scoped_under_the_heading_above_it` |

## Grammar & dependencies

Uses the `tree-sitter-markdown` grammar. No runtime dependency on other
language plugins; the test suite dev-depends on the plugins its sample's
fenced blocks delegate to (bash, css, html, javascript, python, scss — plus
svelte/toml/yaml via core until those are packaged).
