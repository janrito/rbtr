# AGENTS.md

Agent rules. Short, imperative.

---

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
- Simple is better than complex. Use good data structures
  and straightforward APIs.
- Don't build for hypothetical futures. Implement what's
  needed now; extend later when required.
- Every running process must be visible to the user.
  Never hide background work.
- Configuration lives in a central, transparent location.
  No hidden state scattered across the codebase.
- **Persisted history is immutable.** Never modify or inject
  messages into saved conversation history. Repairs and
  adaptations (tool-call pairing, thinking demotion,
  tool-exchange flattening) are applied transiently in memory
  at load time. Record what was changed as incidents — never
  as fake messages.
- Styling is defined centrally. No ad-hoc formatting.

## Style

- Target **Python 3.13+**. Use modern features directly.
- Everything is type-checked and fully annotated.
- Run `just check` after **every** meaningful change.
  Do not batch multiple changes before checking.
- No hacks. Match existing style.
- Never transliterate JS/TS patterns. Rewrite idiomatically.
- `type` aliases for complex annotations.
- `TypedDict` or `BaseModel` for fixed-key dicts.
  No `NamedTuple`.
- **Enums for finite sets.** `Literal` only when an enum
  would be impractical.
- **No `Any` or `object` as lazy type escapes.** Use the
  real type. `Any` only at genuinely untyped boundaries.
  `object` only for Python protocol signatures.
  Never use `cast()` — fix the source.
- Composition over inheritance. Functions over methods
  when in doubt.
- `Protocol` for interfaces. Subclassing only for strict is-a.
- Start with duplication — extract when the pattern is clear.
- All imports at module level. Nested only with a concrete
  reason and a comment explaining why.
- Never embed multi-line code in another language as string
  literals. Write it to its own file and load at runtime.
- Every `# type: ignore` must have a reason on the same line.
- **Markdown-style backticks in docs.** Use single backticks
  (`` ` ``) in docstrings, comments, and documentation.
  Never use rST double-backtick (``` `` ```) literals.
- Fix the root cause first.

## Testing

- **Parametrise over repetition.** One behaviour per test
  function, not one input value.
- **Data-first test design.** Build tests around realistic
  shared datasets. Verify behaviours against concrete data,
  not anonymous stubs.
- **Don't mock internal async iteration protocols.** Test
  through public APIs or mock at the LLM boundary.
- **Pure private functions can be tested directly** when
  doing so avoids mocking the full LLM chain.
- Use library patterns for testing. Don't mock everything.
- **No `unittest.mock`.** Use `pytest-mock` (`mocker` fixture).
- **Fixtures over helpers.** Shared setup belongs in
  `@pytest.fixture`, not in loose helper functions or methods.
- **No test classes.** Plain test functions only.

## Dependencies & tooling

- Prefer maintained libraries over reimplementing.
- Don't install unused dependencies.
- Never change linter, formatter, or tool settings in
  `pyproject.toml` without explicit instructions.
  Fix the code, not the rules.
- **Use `just` recipes for formatting and linting.**
  `just fmt` / `just lint` for everything, or scope by
  language: `fmt-py`, `fmt-sql`, `fmt-md` (and `lint-*`).
  `fmt-md` / `lint-md` accept optional file args
  (default: all). Never run `ruff`, `sqlfluff`, `rumdl`,
  or other tools directly — the recipes carry the correct
  flags and file selections.
