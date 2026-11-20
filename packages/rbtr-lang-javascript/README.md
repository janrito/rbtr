# rbtr-lang-javascript

JavaScript, TypeScript, and TSX support for [rbtr]. A **default**
plugin — installed with rbtr itself (`pip install rbtr`).

[rbtr]: https://github.com/janrito/rbtr/tree/main/packages/rbtr#readme

## What it ingests

One plugin, three ids across two grammars: `javascript` (`.js` / `.mjs` /
`.jsx`), `typescript` (`.ts`), and `tsx` (`.tsx`). A symbol's leading JSDoc is
folded into its chunk content.

- **Functions** — declarations, generators, and arrow functions bound to a
  const; class/object methods (class members scoped to their class).
- **Classes** — classes, and TypeScript interfaces, enums, type aliases, and
  namespaces (a namespace also scopes its members).
- **Variables** — module-level `const` / `let` (including destructuring).
- **Re-exports** — `export * from "./c"`, `export * as ns from "./n"` and
  `export { x, y as z } from "./b"` read from another module, so they resolve
  as imports do. A consumer's calls reach a symbol through them without ever
  naming the file that defines it.
- **Imports** — `import`, `import type`, namespace, default, and side-effect
  imports, and CommonJS `const x = require("./x")` and
  `const { y, z: w } = require("./y")` → import chunks with resolved module +
  names, for cross-file edges. A `require` also stays a variable.
- **Calls** — each function's calls, `new` expressions, and in `.jsx` and
  `.tsx` the components its JSX renders (`<Toolbar>`, `<Ui.Button />`, not
  HTML's `<div>`), for call edges.

## Chunks produced

```js
function greet(name) { … }        // function "greet"
const add = (a, b) => a + b;      // function "add"
class Button { render() {} }      // class "Button"; method "render" scope "Button"
export const MAX = 100;           // variable "MAX"
import { x } from "./x";          // import, metadata {module: "./x", names: "x"}
const { y } = require("./y");     // import {module: "./y", names: "y"}; variable "y"
```

## Embedded / injected chunks

None of its own — JavaScript/TypeScript are embedded *by* the HTML, Markdown,
and SFC (Vue/Svelte) plugins, which delegate `<script>` blocks and fenced code
here.

## Limitations

rbtr resolves calls by the names they are written with, without types
or control flow. These are what that gets wrong in this language; each
is a strict expected failure, so fixing one fails its test.

| What happens                                                                                                                                                                                       | Why                                                                                           | Pinned by                                                  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| A JavaScript default import of an export named otherwise (`import Button from "./widget.js"`, `export default function Widget`) reaches nothing                                                    | It binds its local name, which the module does not define.                                    | `test_a_default_import_reaches_the_default_export`         |
| An import through a re-export (`import { log } from "./index.js"`, where `index.js` has `export { log } from "./util.js"`) makes no import edge to `util.js`'s `log`; calls through it do reach it | An import edge reaches the definitions in the module it names, and `index.js` defines nothing | `test_an_import_through_a_reexport_reaches_the_definition` |

## Grammar & dependencies

Uses the `tree-sitter-javascript` and `tree-sitter-typescript` grammars. No
runtime dependency on other language plugins; the test suite dev-depends on
`rbtr-lang-css` so the sample's `styles.css` extracts and the cross-file edges
snapshot.
