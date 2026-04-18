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
- **All imports at module level.** Deferred (nested) imports
  are almost never justified. Valid reasons: heavy native
  libraries that add measurable import latency (`tree_sitter`,
  `llama_cpp`), or breaking a genuine circular import that
  cannot be restructured. Every deferred import must have a
  `# deferred: <reason>` comment. "Might be slow" or "avoid
  importing too much" are not valid reasons — measure first.
- Never embed multi-line code in another language as string
  literals. Write it to its own file and load at runtime.
- **No implicit string concatenation.** Use `\` line
  continuation or triple-quoted strings for long strings.
  Never rely on adjacent-literal concatenation.
- **`# type: ignore` is a last resort.** Fix the type first.
  Use the correct generic parameters, narrow with `isinstance`,
  or restructure the code. Only suppress when the type system
  genuinely cannot express the situation (e.g. pydantic
  `__init__` re-init, untyped third-party libraries). Every
  suppression must include the error code and a reason:
  `# type: ignore[code]  # why`.
- **Markdown-style backticks in docs.** Use single backticks
  (`` ` ``) in docstrings, comments, and documentation.
  Never use rST double-backtick (` `` `) literals.
- **Reference-style links for long URLs.** When an inline
  link would exceed the line-length limit, use a markdown
  reference link with the definition immediately after the
  paragraph that uses it.
- Fix the root cause first.

## Testing

- **Red/green TDD.** Write a failing test first, then write
  the code to make it pass. Run `just check` after each step.
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
  That includes private functions at module scope called only
  from a fixture or a case — if a fixture body is too long,
  decompose it into *smaller fixtures*, not helper functions.
- **Prefer independent fixtures to factory fixtures.** A fixture
  that returns a callable (a "factory") hides dependencies behind
  invocation: the dependency graph pytest builds becomes opaque,
  and fixtures can be called repeatedly or not at all without the
  test making that explicit. When the resources a test needs are
  independent, declare one fixture per resource and let pytest
  compose them. Nested dependent fixtures (fixture A takes
  fixture B) are fine — they still expose the graph. Factories
  are a smell; reach for one only when the test truly needs many
  instances parametrised by caller-supplied arguments.
- **Test data and setup live in fixtures.** Setup is setup —
  whether it is an action or a value. Module-level constants
  (including private `_FOO`) and module-level setup helpers
  are both bad: they hide dependencies (no fixture parameter
  makes them explicit), cannot be parametrised, and encourage
  tests to import directly instead of declaring what they
  consume.

  Rules:
  1. No module-level constants used by tests or cases.
     Even pure frozen values belong in fixtures.
  2. No module-level *setup* functions used by tests or
     cases. Setup = anything that constructs something the
     test didn't already have: chunks, stores, repos, mocks,
     fake services. Long fixture bodies decompose into
     *smaller fixtures*, never into setup helpers.
  3. **Pure projections over test-visible data are allowed.**
     Small module-level functions that transform values the
     test *already has* (e.g. `rank(results, chunk_id) ->
     int` or `ids(results) -> list[str]`) are fine. Heuristic:
     does it construct anything the test didn't ask for?
     If yes, fixture. If no, inline it or keep it as a
     plain helper. These helpers do not hide dependencies,
     because the data they operate on arrived via fixture
     parameters of the test itself.
  4. Shared values go into `conftest.py` fixtures so pytest
     resolves them positionally.
  5. The single exception for constants is values the
     *production code* also uses (e.g. `SCHEMA_VERSION`
     imported for an assertion). These are not test data —
     they are the production surface under test.
  6. **Parametrize-time values.** Everything pytest needs at
     collection time (before any fixture runs) may be module
     level: `pytest.mark.skipif` markers passed via
     `pytest.param(..., marks=...)`, and the argument lists /
     dicts consumed by `@pytest.mark.parametrize(...)`.
     Fixtures cannot supply these values. Prefer inline
     literals inside the decorator when practical; when the
     data is large enough that inlining hurts readability, a
     module-level `_NAME = [...]` is acceptable. This is
     pytest architecture, not a style choice.

  "It would add ceremony" is not a justification; one
  `@fixture` line per value is not ceremony. If a rewrite
  touches many files, do the rewrite — file count is not a
  design argument.

- **Bias against abstraction in tests.** The point of a test
  is to exercise the widest sensible slice of production code
  against concrete data. Helpers, abstractions, and
  parametrisation are tools; they are not goals. Prefer
  explicit, data-first scenarios even when they repeat a few
  lines.
- **No test classes.** Plain test functions only.

## Git

- **Never rewrite history.** No `git commit --amend`,
  no `git rebase`, no `git push --force`. Every commit
  is final. If a commit is wrong, fix it in a new commit.
  Amending destroys context and makes the log a lie.

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
- **Don't lint TODO files.** `TODO-*.md` files are planning
  documents with tables and free-form prose. Don't run
  `just lint-md` on them or try to fix their warnings.
- **Never `git add -f` gitignored paths.** If a path is in
  `.gitignore`, it stays out of version control. No
  exceptions without explicit instruction.
- **TODO plans are ephemeral.** Never reference phase
  numbers, plan names, or TODO file contents in commit
  messages, code comments, docstrings, or documentation.
  Those are work-in-progress scratchpads that won't live
  next to the code. Commit messages describe *what changed
  and why*, not which plan step was executed.
