# AGENTS.md

Agent rules. Short, imperative.

---

## Project

- The project is **rbtr** — always lowercase.
- rbtr is a **language-agnostic** code review tool.
  Never hard-code behaviour for a single language when a
  general mechanism exists.

## Architecture

- The engine (`engine/` package, entry point `engine/core.py`) and
  UI (`tui.py`) communicate through `queue.Queue[Event]`.
- Engine never imports Rich. UI never runs commands or does I/O.
- `input.py` is a pure utility module — no Rich, no Engine.
  Safe to call from any thread.
- Tasks run in daemon threads via `_start_task()` (in TUI).
  Never block the main thread.
- New output types → new event in `events.py`, handle in
  both Engine and UI.

## Code index & plugins

- Language support is provided by **plugins** via pluggy.
  Each plugin registers `LanguageRegistration` instances
  declaring detection, grammar, queries, import extraction,
  and scope types.
- `index/treesitter.py` and `index/edges.py` are language-agnostic.
  All language-specific logic lives in `plugins/`.
  Adding a language = adding a plugin file.
- Use tree-sitter for structural analysis when a grammar
  provides the data. When it doesn't (unsupported language,
  prose docs, config files), fall back to text search —
  never silently skip.
- Index storage uses DuckDB (`index/store.py`) with SQL files
  in `index/sql/`. Embeddings via llama-cpp-python (`index/embeddings.py`).
- Tests must cover multiple languages, not just Python.
  Use `pytest.mark.skipif` for optional grammar packages.

## Engine: input & commands

- Plain text → LLM. `/command` → internal handler. `!command` → shell.
- Commands are a `Command` enum (`engine/types.py`), dispatched
  via `match`/`case` in `_handle_command()`.
- `InputReader` runs in a daemon thread. `InputState` is
  shared state between reader and main loop.
- Enter submits. Alt+Enter inserts newline. Tab sets
  `tab_pressed` flag for the main loop.
- History persists to disk on every submit.
- Shell completion (`!commands`): three tiers —
  bash programmable → filesystem → PATH executables.
- Slash completion (`/commands`): completes command names
  and arguments (e.g. `/connect` providers, `/model` IDs).
- Completion runs in background threads. Pure functions in `input.py`.

## Engine: shell & cancellation

- Child processes get a pty slave as stdin
  (reader thread owns real stdin exclusively).
- `start_new_session=True`. Save PGID immediately after creation.
- `communicate(timeout=0.02)` in a loop with cancel checks.
- `_out()` calls `_check_cancel()` automatically.
- `on_cancel` fires from the reader thread — sends SIGTERM
  to saved PGID immediately.
- `TaskCancelled` cleanup sends SIGKILL via saved PGID.

## Engine: providers & conversation

- Each provider module (`providers/claude.py`, `providers/openai.py`,
  `providers/openai_codex.py`, `providers/endpoint.py`) owns
  auth flow + model construction.
- Shared OAuth/PKCE utilities in `oauth.py` (package root).
  Helper functions: `oauth_is_set()`, `oauth_expired()`.
- `OAuthCreds` (in `creds.py`) is a plain data model.
- Model IDs use `<provider>/<model-id>` format everywhere
  (e.g. `claude/claude-sonnet-4-20250514`).
- `providers/__init__.py` dispatches `build_model()` by
  provider prefix (`BuiltinProvider` enum).
- **Always preserve history across model and provider switches.**
  PydanticAI's message format is provider-agnostic.
  Only `/new` clears history (explicit user action).

## UI & styling

- Rich owns all rendering. No Textual, curses, or raw ANSI escape codes.
- prompt_toolkit is used headlessly — `Vt100Parser` for
  keystroke parsing, `Buffer` for editing state.
  It must never write to the terminal.
- All history blocks go through `_history_panel(variant, content)`.
  No ad-hoc `Padding(...)`.
- Panel variants: `input`, `active`, `succeeded`, `failed`, `queued`.
- Before printing to scrollback via `console.print()`,
  call `_live.update(self._render_view(), refresh=True)`
  so Live shrinks first.
- All styling in `styles.py` via `rich.theme.Theme`.
  Palette is Ayu Mirage.
- Theme keys use `rbtr.*` namespace. Hex values only inside `THEME`.
- Engine references styles via exported string constants,
  never Rich imports.

## Configuration & credentials

- Config: `config.py` — layered TOML
  (class defaults → `~/.config/rbtr/config.toml` →
  `.rbtr/config.toml`) via pydantic-settings.
  Singleton `config`, mutate via `config.update(**kwargs)`.
- Credentials: `creds.py` — `~/.config/rbtr/creds.toml`, 0600.
  Singleton `creds`, mutate via `creds.update(**kwargs)`.
- `update()` builds a merged copy, persists, then calls
  `__init__()` to reload in place. Identity never changes —
  direct imports are safe.
- Endpoint URLs live in config, API keys in creds.
- `RBTR_DIR` (base path `~/.config/rbtr`) in `constants.py`.
  Module-specific paths (`CONFIG_PATH`, `CREDS_PATH`,
  `HISTORY_PATH`) live in the modules that own them.
- **Module-level constants vs config fields.** Before adding a
  magic number as a module constant, ask: would a user or a
  different repo ever want a different value? If yes → config
  field. If truly internal (protocol limits, retry counts) →
  constant. Tunables that affect output (line limits, char
  limits, timeouts, thresholds) almost always belong in config.

## Python

- Target **Python 3.13+**. Use modern features directly.
- Run `just check` after **every** meaningful change —
  especially immediately after any behavioral change
  (refactoring, extraction, moving code). Do not batch
  multiple changes before checking.
- No hacks. Match existing style.
- When using JS/TS code as reference, understand the intent
  and rewrite idiomatically in Python. Never transliterate
  JS patterns — no callback pyramids, no `Promise`-shaped code,
  no `null`-checks where `Optional` and pattern matching work,
  no hand-rolled retry/event loops when stdlib or library
  equivalents exist.

## Code style

- Prefer `match`/`case` over `if`/`elif` chains.
- Name complex annotations with `type` aliases.
- `TypedDict` or `BaseModel` for fixed-key dicts. No `NamedTuple`.
- **Enums for finite sets.** When a variable accepts a known set
  of values, define a `StrEnum` or `IntEnum`. Use `Literal` only
  when an enum would be impractical (e.g. values from a 3rd-party API).
- **No `Any` or `object` as lazy type escapes.** Use the real type.
  When a library returns a concrete type, import and use it
  (e.g. `ModelMessage`, `ModelSettings`, `FrameType`).
  `Any` is acceptable only for genuinely untyped boundaries
  (JSON dicts from a DB driver, TOML `**kwargs`).
  `object` is acceptable only for Python protocol signatures
  (e.g. `_missing_(cls, value: object)`, `__exit__(*args: object)`).
  Never use `cast()` to paper over a type mismatch — fix the source.
- Composition over inheritance. Functions over methods when in doubt.
- `Protocol` for interfaces. Subclassing only for strict is-a
  (e.g. exceptions).
- Start with duplication — extract only when the pattern is clear.
- All imports at module level by default.
  Nested imports **only** with a concrete reason (circular dep, heavy
  module, conditional availability) and a comment explaining why.
- Never embed multi-line code in another language as string literals
  inside Python. Write it to its own file and load at runtime via
  `importlib.resources` or a path constant.
  Exception: one-liners or very short commands.

## Errors & suppressions

- Raise `RbtrError`. No `sys.exit()` in library code.
- No bare `except Exception` — catch the narrowest type.
  Only acceptable at outermost Engine task boundary.
- Use `http.HTTPStatus` for status checks.
- Always specify exact suppression codes
  (`# noqa: S602`, `# type: ignore[assignment]`).
- Every `# type: ignore` **must** have a short reason on the
  same line (e.g. `# type: ignore[misc]  # pydantic __init__ reload`).
- Fix the root cause first.

## Testing

- Default: test via Engine event contract.
- pytest-mock only (`mocker` fixture). Never `unittest.mock`.
- Flat `def test_*()` functions. No `class Test*`.
- `@pytest.fixture` without parentheses.
- Tests live in `src/tests/`, mirroring package structure.
- Shared fixtures in `src/tests/conftest.py`.
- **Never skip based on the return value of the function under test.**
  Detect environment capabilities separately and use
  `pytest.mark.skipif`.
- **Parametrise over repetition.** Use `@pytest.mark.parametrize`
  when multiple tests share the same assertion logic.
  Each test function should test one *behaviour*, not one *input value*.
- **Multiline strings for test source code.** Use `"""\..."""` for
  any inline code snippet longer than one line — never concatenate
  strings or use escaped `\n`.
- **Data-first test design.** Build test suites around a shared,
  realistic dataset — not around the API surface.
  Define semantically distinct test entities (e.g. three functions
  from different domains with orthogonal embedding vectors) as
  module-level constants. Seed them through a helper
  (e.g. `_seed_store(store, embed=True)`).
  Then write tests that verify *behaviours against this data*:
  ranking, scoping, edge cases, roundtrips.
  This grounds every assertion in concrete, inspectable content
  instead of anonymous stubs or mocks.
  When the data is right, the tests document the system.

## Dependencies & tooling

- Use high-quality, maintained libraries over reimplementing.
- Don't install dependencies that are not used.
- httpx for HTTP. Never urllib or requests.
- Never change linter, formatter, or tool settings in
  `pyproject.toml` without explicit instructions.
  When lint rules flag issues, fix the code — don't loosen the rules.
