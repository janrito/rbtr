<!--
  Svelte single-file component.

  The SFC plugin delegates the <script> block to TypeScript and the
  <style> block to SCSS; their symbols carry those languages and their
  imports resolve through the matching resolver. The markup template is
  not extracted.
-->
<script lang="ts">
  import { greet } from "./greeter";

  export let name: string = "world";

  function shout(): string {
    return greet(name).toUpperCase();
  }
</script>

<style lang="scss">
  $accent: #00aa77;

  .greeting {
    color: $accent;
    font-weight: bold;
  }
</style>

<h1 class="greeting">{shout()}</h1>
