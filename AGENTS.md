# AGENTS.md

## Project

- The project is **rbtr** — always lowercase.
- rbtr is a **language-agnostic** code review tool.
  Never hard-code behaviour for a single language when a
  general mechanism exists.

## Architecture

- Components must be independent, communicating through
  defined APIs. Don't mix concerns.
- Data must be saved transactionally. Design for recovery
  from failure at any point.
- Never block the UI. Use threads for all workloads.
- Every running process must be visible to the user.
  Never hide background work.
- Configuration lives in a central, transparent location.
  No hidden state scattered across the codebase.
- Styling is defined centrally. No ad-hoc formatting.

## Style

- Target **Python 3.13+**. Use modern features directly.
- Everything is type-checked and fully annotated.
- Run `just check` after **every** meaningful change.
  Do not batch multiple changes before checking.
- No hacks. Match existing style.
- Never transliterate JS/TS patterns. Rewrite idiomatically.
- Fix the root cause first.

### Types & annotations

- `type` aliases for complex annotations.
- `TypedDict` or `BaseModel` for fixed-key dicts.
  No `NamedTuple`.
- **Enums for finite sets.** `Literal` only when an enum
  would be impractical.
- **No `Any` or `object` as lazy type escapes.** Use the
  real type. `Any` only at genuinely untyped boundaries.
  `object` only for Python protocol signatures.
  Never use `cast()` — fix the source.
- **`# type: ignore` is a last resort.** Fix the type first.
  Only suppress when the type system genuinely cannot express
  the situation. Every suppression must include the error code
  and a reason: `# type: ignore[code]  # why`.

### Design

- Composition over inheritance. Functions over methods
  when in doubt.
- `Protocol` for interfaces. Subclassing only for strict is-a.
- Start with duplication — extract when the pattern is clear.

### Imports & strings

- **All imports at module level.** Deferred imports only for
  heavy native libraries (`tree_sitter`, `llama_cpp`) or
  genuine circular imports. Each must have a
  `# deferred: <reason>` comment.
- **No re-exports.** Never write `from x import y` then
  expose `y` from a different module. Each symbol has one
  home. Consumers import from the source.
- Never embed multi-line code in another language as string
  literals. Write it to its own file and load at runtime.
- **No implicit string concatenation.** Use `\` line
  continuation or triple-quoted strings.
- **Markdown-style backticks in docs.** Single backticks
  (`` ` ``), never rST double-backtick.
- **Reference-style links for long URLs.**
- **No block comments duplicating a docstring.**

### Documentation split

Three layers, each with a distinct audience and purpose.
Every piece of prose belongs in exactly one.

- **README** — for users and adopters. What this is,
  how to install it, how to use it. Show, don't tell:
  examples with realistic output. No implementation
  detail. Self-contained — a reader landing from a
  search engine shouldn't need another file first.
- **ARCHITECTURE.md** — for contributors changing the
  system. How components connect, why decisions were
  made, data flows, concurrency models. Cross-module
  interactions and design rationale live here.
- **Docstrings** — for callers of this specific symbol.
  What it accepts, what it returns, what invariants it
  maintains. Contract, not narrative.

Boundary rules:

- **Algorithms** belong in the function's docstring if
  they affect how callers use it. They belong in
  ARCHITECTURE if they're about system-level
  coordination.
- **Cross-module interactions** belong in ARCHITECTURE.
  A docstring should not narrate its own call chain.
- **Design rationale** belongs in ARCHITECTURE's
  design-decisions section, not in docstrings or README.
- **Historical context** belongs in none of the three.
  Describe the present. History lives in git.
- **Edge-case caveats** about implementation limits
  belong as inline `#` comments, not in docstrings.
- **Each concept once.** Other mentions cross-reference
  with a link. Never duplicate an explanation across
  README, ARCHITECTURE, and docstrings.

### Library usage

- **Check the library before writing a util.** `rbtr`, the
  stdlib, and already-imported libraries likely have what you
  need. Private helpers are a cost — don't add one you don't
  need.
- **Use `platformdirs` via `Config.*_path`.** Never hardcode
  `~/.foo` or `$HOME/.foo`. See `rbtr.config` module docstring
  for the full directory layout.

### Data & testing

Data handling conventions are in the **`rbtr-data`** skill.
Testing conventions are in the **`rbtr-testing`** skill.

## Git

- **Never rewrite history.** No `git commit --amend`,
  no `git rebase`, no `git push --force`. Every commit
  is final. If a commit is wrong, fix it in a new commit.
- **Commit messages are short.** One summary sentence;
  optional body of ≤ 5 sentences. Describe _what_ changed
  and _why_. Never reference TODO files or plan content.

## Dependencies & tooling

- Prefer maintained libraries over reimplementing.
- Don't install unused dependencies.
- **Use `just` recipes for all CI tasks.**
  `just fmt` / `just lint` (or `fmt-py`, `fmt-sql`, `fmt-md`
  and `lint-*`). `just test` for the full suite;
  `just typecheck` for types; `just check` for the complete
  gate. Never run tools directly.
- **Don't lint TODO files.** They are planning documents.
- **Never `git add -f` gitignored paths.**
