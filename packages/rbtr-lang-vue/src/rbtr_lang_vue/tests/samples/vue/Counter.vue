<!--
  Vue single-file component.

  The SFC plugin delegates <script> to TypeScript and <style> to SCSS.
  This comment is itself a chunk, so a component's own explanation of
  what it is for stays searchable.
-->

<script setup lang="ts">
// Vue single-file component.
//
// The SFC plugin delegates the <script setup> block to TypeScript and
// the <style> block to SCSS, reusing the same chunker as Svelte. The
// template is not extracted.
import { makeStore } from "./store";

const store = makeStore();

function increment(): void {
  store.count += 1;
}
</script>

<style lang="scss">
  $accent: #00aa77;

  .counter {
    color: $accent;
    font-weight: bold;
  }
</style>

<template>
  <button class="counter" @click="increment">{{ store.count }}</button>
</template>
