# rbtr-lang-html

HTML support for [rbtr](../rbtr). A **default** plugin — installed with rbtr
itself (`pip install rbtr`).

## What it ingests

- **Sections** — major `<body>` elements (`<main>`, `<article>`, `<nav>`, …)
  → doc sections, named by their `id` (else the tag name).
- **Imports** — `<script src>` and `<link href>` → import chunks carrying the
  referenced path and a language hint, so edges link to the target file.

## Chunks produced

```html
<main id="content">…</main>           <!-- doc_section "content"                  -->
<nav>…</nav>                          <!-- doc_section "nav"                      -->
<script src="app.js"></script>        <!-- import {module: app.js, hint: javascript} -->
<link rel="stylesheet" href="a.css">  <!-- import {module: a.css, hint: css}      -->
```

## Embedded / injected chunks

HTML embeds other languages: an inline `<script>` delegates to JavaScript and
an inline `<style>` to CSS, so their chunks extract at real line numbers
within the page, each carrying its own language.

```html
<main id="app">                    <!-- doc_section "app"   (html)       -->
  <script>
    function boot() { start(); }   <!-- function "boot"     (javascript) -->
  </script>
  <style>
    .hero { color: crimson; }      <!-- class ".hero"       (css)        -->
  </style>
</main>
```

## Grammar & dependencies

Uses the `tree-sitter-html` grammar. No runtime dependency on other language
plugins; the test suite dev-depends on `rbtr-lang-javascript` and
`rbtr-lang-css` so the sample's linked and injected js/css extract and the
cross-file edges snapshot.
