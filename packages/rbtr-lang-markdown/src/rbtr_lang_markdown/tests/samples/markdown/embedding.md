# Embedding

Examples embedded in the docs, each extracted in its own language —
including code nested two levels deep.

The greeter config, in YAML:

```yaml
service: greeter
locales:
  - en
  - fr
```

The same in TOML:

```toml
[service]
name = "greeter"
```

Embedded in a page — the stylesheet is an import and the inline script is
extracted as JavaScript:

```html
<head>
  <link rel="stylesheet" href="greeter.css">
</head>
<body>
  <main>
    <script>
      function mount() {
        return greet("Ada");
      }
    </script>
  </main>
</body>
```

And a component — Markdown delegates to the Svelte chunker, which in turn
delegates its `<script>` to TypeScript and its `<style>` to SCSS:

```svelte
<script lang="ts">
  export let name: string;

  function shout(): string {
    return name.toUpperCase();
  }
</script>

<h1>Hello {name}</h1>

<style lang="scss">
  h1 {
    color: $brand;
  }
</style>
```
