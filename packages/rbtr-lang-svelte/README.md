# rbtr-lang-svelte

Svelte support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[svelte]`. Also hosts the shared single-file-component
machinery that [rbtr-lang-vue](../rbtr-lang-vue) reuses.

## What it ingests

`.svelte` components. A component mixes several languages in one file, and
each part is indexed as itself:

- The **markup template** becomes one host **doc-section** chunk (named after
  the file), so component markup is searchable.
- The `<script>` and `<style>` blocks are **delegated** to their embedded
  language by injection (see below) — not re-parsed here.

## Chunks produced

```svelte
<script lang="ts">
  import { store } from "./store";     // → import        (typescript)
  export let name = "world";           // → variable "name" (typescript)
</script>

<h1 class="greeting">Hello {name}</h1> <!-- → doc_section "Greeting" (svelte host) -->

<style lang="scss">
  .greeting { color: red; }            // → class ".greeting" (scss)
</style>
```

## Embedded / injected chunks

The `lang` attribute picks the dialect; a bare block uses the fallback:

- `<script lang="ts">` → **TypeScript**, bare `<script>` → **JavaScript**
- `<style lang="scss">` → **SCSS**, `lang="less"` → **Less**, bare `<style>`
  → **CSS**

These chunks carry their *embedded* language; the delegated extraction runs
only when the corresponding plugin (e.g. `rbtr-lang-javascript`,
`rbtr-lang-css`) is installed.

## Grammar & dependencies

Uses the `tree-sitter-svelte` grammar. Ships `chunk_sfc` and `injections.scm`,
which `rbtr-lang-vue` imports for `.vue` files (they share the SFC shape).
