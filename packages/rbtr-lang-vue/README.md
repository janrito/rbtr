# rbtr-lang-vue

Vue support for [rbtr](../rbtr). Optional plugin — install with
`pip install rbtr[vue]`.

## What it ingests

`.vue` single-file components. Vue has the same `<script>`/`<style>`/template
shape as Svelte, so this package reuses the SFC chunker and injection query
from [rbtr-lang-svelte](../rbtr-lang-svelte): the markup template becomes one
host **doc-section** chunk, and the `<script>`/`<style>` blocks are delegated
to their embedded language.

## Chunks produced

```vue
<script setup lang="ts">
import { store } from "./store"     // → import        (typescript)
const count = ref(0)                // → variable "count" (typescript)
</script>

<template>
  <button>{{ count }}</button>      <!-- → doc_section "Counter" (vue host) -->
</template>

<style lang="scss">
.btn { color: red; }                // → class ".btn" (scss)
</style>
```

## Embedded / injected chunks

`<script lang="ts">` → TypeScript (bare → JavaScript); `<style lang="scss">` →
SCSS (`lang="less"` → Less, bare → CSS). These run only when the corresponding
plugin (`rbtr-lang-javascript`, `rbtr-lang-css`) is installed.

## Grammar & dependencies

The Vue grammar is bundled in `tree-sitter-language-pack`. Depends on
`rbtr-lang-svelte` for the shared SFC chunker and injection query.
