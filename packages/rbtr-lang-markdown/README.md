# rbtr-lang-markdown

Markdown support for [rbtr](../rbtr). A **default** plugin — installed with
rbtr itself (`pip install rbtr`).

## What it ingests

A custom chunker (not a tree-sitter query): it splits a document by its
**heading hierarchy** — each section is a chunk, scoped by its ancestor
headings. Links and fenced code blocks are extracted on top.

- **Sections** — headings → doc sections; `scope` is the `::` chain of
  enclosing headings. A headingless document falls back to plaintext chunks.
- **Links** — local `[text](path.md)` and `[text](path.md#fragment)` →
  imports (the fragment becomes `names`). External and fragment-only links
  are skipped.
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
