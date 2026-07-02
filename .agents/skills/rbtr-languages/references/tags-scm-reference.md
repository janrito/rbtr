# `tags.scm` reference (curated)

The grammar authors' `queries/tags.scm` is the canonical "definitions and
references" query behind `tree-sitter tags` / GitHub code navigation. We **mine
it for inspiration** — take good patterns, adapt weak ones, ignore wrong ones —
we never run it live (see the skill body for why). Verify every pattern against
a real parse before adopting; quality is uneven.

**Where to find it.** In each installed grammar package:
`tree_sitter_<lang>/queries/tags.scm` (resolve with
`importlib.import_module("tree_sitter_<lang>").__file__`). Upstream:
`github.com/tree-sitter/tree-sitter-<lang>` (some, e.g. sql/hcl, ship none).

**Standard vocabulary.** `@name`, `@definition.{class,function,method,
interface,type,module,macro,constant}`, `@reference.{call,type,class,
implementation}`.

**Mapping to rbtr `ChunkKind`** (the spine of any pattern we adopt):

| tags capture            | rbtr kind  | note                                                    |
| ----------------------- | ---------- | ------------------------------------------------------- |
| `@definition.class`     | `class`    |                                                         |
| `@definition.function`  | `function` |                                                         |
| `@definition.method`    | `method`   |                                                         |
| `@definition.interface` | `class`    | traits, interfaces                                      |
| `@definition.type`      | `class`    | type aliases / typedefs (Question 1)                    |
| `@definition.module`    | `class`    | namespace/module as a symbol (low priority)             |
| `@definition.macro`     | `function` | `macro_rules!`, `#define` function-macros               |
| `@definition.constant`  | `variable` |                                                         |
| `@reference.*`          | —          | usage, not a definition; candidate for CALLS/edges only |

Caveats that apply everywhere: **no imports**, **few variables** — those stay
bespoke. References are noise for symbol extraction.

## Per-language verdicts

### rust — TAKE (high value; closes every current gap)

`tree_sitter_rust/queries/tags.scm`. Captures the constructs our query misses.
Adopt these node types (verified):

```scm
(trait_item name: (type_identifier) @name) @definition.interface   ; -> class
(macro_definition name: (identifier) @name) @definition.macro       ; -> function
(mod_item name: (identifier) @name) @definition.module              ; -> class/scope
```

Also `type_item` and `union_item` (→ class). Add `trait_item` to
`scope_types`/`class_scope_types` so trait methods scope + promote.

### typescript — TAKE (partial) + SUPPLEMENT

`tree_sitter_typescript/queries/tags.scm`. Take:

```scm
(interface_declaration name: (type_identifier) @name) @definition.interface
(abstract_class_declaration name: (type_identifier) @name) @definition.class
```

(plus `internal_module`/`module` for namespaces, and `method_signature` /
`abstract_method_signature` for declared methods). **Missing — supplement
ourselves**: `enum_declaration`, `type_alias_declaration`. Note it may inherit
javascript's tags; our base query already covers plain class/function.

### java — TAKE (partial) + SUPPLEMENT

`tree_sitter_java/queries/tags.scm`. Take `interface_declaration`
(`@definition.interface` → class). **Missing — supplement**:
`enum_declaration`, `record_declaration`. Add all three to `scope_types` so
their members scope correctly.

### go — small win

`tree_sitter_go/queries/tags.scm`. `@definition.type` covers both `type_spec`
and `type_alias`, so it resolves our type-alias gap:

```scm
(type_declaration (type_spec name: (type_identifier) @name)) @definition.type
```

### c — IGNORE the union pattern (it's wrong for us)

`tree_sitter_c/queries/tags.scm`. Its union pattern only matches a union in a
*declaration*, not the common standalone definition:

```scm
; tags.scm — MISSES `union U { int a; };`
(declaration type: (union_specifier name: (type_identifier) @name)) @definition.class
```

Write our own body-required pattern instead:

```scm
(union_specifier name: (type_identifier) @_cls_name body: (field_declaration_list)) @class
```

Also note tags maps `enum_specifier` → `@definition.type`; we map enum → class
for consistency with struct.

### cpp — mostly already covered

`tree_sitter_cpp/queries/tags.scm`. Has `@definition.{class,function,method,
type}`. Marginal over our query; consult for `using`/alias and namespace
patterns if adding them.

### python / javascript / ruby — little to mine

Their tags.scm (`@definition.class`/`function`/`method`/`constant`, plus
ruby `module`) is already matched by our queries. No action; consult only if a
specific construct is reported missing.

## Workflow for adopting a pattern

1. Open the grammar's `tags.scm`; find the relevant `@definition.*` pattern.
2. Parse a real snippet of the construct; confirm the node types/fields match
   (grammars drift; tags.scm can be narrow — see C union).
3. Translate to our capture convention (`@_cls_name`/`@class` etc.), adding
   `body:`/anchors as needed to match definitions only.
4. Add to the sample, `just snapshots`, bump version, `just check`.
