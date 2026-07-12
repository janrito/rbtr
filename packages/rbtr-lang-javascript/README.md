# rbtr-lang-javascript

JavaScript, TypeScript, and TSX support for [rbtr](../rbtr). A **default**
plugin — installed with rbtr itself (`pip install rbtr`).

## What it ingests

One plugin, three ids across two grammars: `javascript` (`.js` / `.mjs` /
`.jsx`), `typescript` (`.ts`), and `tsx` (`.tsx`). A symbol's leading JSDoc is
folded into its chunk content.

- **Functions** — declarations, generators, and arrow functions bound to a
  const; class/object methods (class members scoped to their class).
- **Classes** — classes, and TypeScript interfaces, enums, type aliases, and
  namespaces (a namespace also scopes its members).
- **Variables** — module-level `const` / `let` (including destructuring).
- **Imports** — `import`, `import type`, namespace, default, and side-effect
  imports → import chunks with resolved module + names, for cross-file edges.

## Chunks produced

```js
function greet(name) { … }        // function "greet"
const add = (a, b) => a + b;      // function "add"
class Button { render() {} }      // class "Button"; method "render" scope "Button"
export const MAX = 100;           // variable "MAX"
import { x } from "./x";          // import, metadata {module: "./x", names: "x"}
```

## Embedded / injected chunks

None of its own — JavaScript/TypeScript are embedded *by* the HTML, Markdown,
and SFC (Vue/Svelte) plugins, which delegate `<script>` blocks and fenced code
here.

## Grammar & dependencies

Uses the `tree-sitter-javascript` and `tree-sitter-typescript` grammars. No
runtime dependency on other language plugins; the test suite dev-depends on
`rbtr-lang-css` so the sample's `styles.css` extracts and the cross-file edges
snapshot.
