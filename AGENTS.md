# AGENTS.md

Agent rules. Short, imperative.

## Naming

- The project is **rbtr** — always lowercase.

## Architecture

- Engine (`engine.py`) and UI (`tui.py`) communicate
  through `queue.Queue[Event]`.
- Engine never imports Rich. UI never runs commands or does I/O.
- `input.py` is a pure utility module — no Rich, no Engine.
  Safe to call from any thread.
- Tasks run in daemon threads via `_start_task()`. Never block the main thread.
- New output types → new event in `events.py`, handle in both Engine and UI.

## UI

- Rich owns all rendering. No Textual, curses, or raw ANSI escape codes.
- prompt_toolkit is used headlessly — `Vt100Parser` for keystroke parsing,
  `Buffer` for editing state. It must never write to the terminal.
- All history blocks go through `_history_panel(variant, content)`.
  No ad-hoc `Padding(...)`.
- Panel variants: `input`, `active`, `succeeded`, `failed`, `queued`.
- Before printing to scrollback via `console.print()`,
  call `_live.update(self._render_view(), refresh=True)` so Live shrinks first.

## Input

- `InputReader` runs in a daemon thread. `InputState` is
  shared state between reader and main loop.
- Enter submits. Alt+Enter inserts newline. Tab sets
  `tab_pressed` flag for the main loop.
- History persists to disk on every submit.

## Completion

- Shell completion (`!commands`): three tiers —
  bash programmable → filesystem → PATH executables.
- Slash completion (`/commands`): completes command names
  and arguments (e.g. `/connect` providers, `/model` IDs).
- Runs in background threads. Pure functions in `input.py`.

## Styling

- All styling in `styles.py` via `rich.theme.Theme`. Palette is Ayu Mirage.
- Theme keys use `rbtr.*` namespace. Hex values only inside `THEME`.
- Engine references styles via exported string constants, never Rich imports.

## Configuration & credentials

- Config: `cfg.py` — layered TOML
  (package defaults → user → workspace) via pydantic-settings.
  Singleton ``cfg``, mutate via ``cfg.update(**kwargs)``.
- Credentials: `creds.py` — `~/.config/rbtr/creds.toml`, same pattern, 0600.
  Singleton ``creds``, mutate via ``creds.update(**kwargs)``.
- ``update()`` builds a merged copy, persists, then calls ``__init__()``
  to reload in place. Identity never changes — direct imports are safe.
- OAuth credentials use ``OAuthCreds`` everywhere (including providers).
  ``OAuthCreds`` has ``.expired`` and ``.is_set`` properties.
- Endpoint URLs live in config, API keys in creds.
- Constants (`CONFIG_PATH`, `CREDS_PATH`, `HISTORY_PATH`) in `constants.py`.

## Providers

- Each provider module owns auth flow + model construction.
- Shared OAuth/PKCE utilities in `providers/oauth.py`.
- Model IDs use `<provider>/<model-id>` format everywhere
  (e.g. `claude/claude-sonnet-4-20250514`).
- `providers/__init__.py` dispatches `build_model()` by provider prefix.

## Conversation history

- **Always preserve history across model and provider switches.**
  PydanticAI's `ModelRequest`/`ModelResponse` message format is
  provider-agnostic — the library handles conversion when sending
  to each provider. Never clear history on `/model` change.
- Only `/new` clears history (explicit user action).

## Shell execution (`!commands`)

- Child processes get a pty slave as stdin (reader thread owns real stdin exclusively).
- `start_new_session=True`. Save PGID immediately after creation.
- `communicate(timeout=0.02)` in a loop with cancel checks.

## Cancellation

- `_out()` calls `_check_cancel()` automatically.
- `on_cancel` fires from the reader thread — sends SIGTERM to saved PGID immediately.
- `_TaskCancelled` cleanup sends SIGKILL via saved PGID.

## Input routing

- Plain text → LLM. `/command` → internal handler. `!command` → shell.
- Commands in `_COMMANDS` dict. Keep commands minimal.

## Errors

- Raise `RbtrError`. No `sys.exit()` in library code.
- No bare `except Exception` — catch the narrowest type.
  Only acceptable at outermost Engine task boundary.
- Use `http.HTTPStatus` for status checks.

## Dependencies

- Use high-quality, maintained libraries over reimplementing generic functionality.
- Don't install dependencies that are not used.
- httpx for HTTP. Never urllib or requests.

## Testing

- Default: test via Engine event contract.
- pytest-mock only (`mocker` fixture). Never `unittest.mock`.
- Flat `def test_*()` functions. No `class Test*`.
- `@pytest.fixture` without parentheses.
- Shared fixtures in `conftest.py` (`creds_path`, `config_path`).
- **Never skip based on the return value of the function under test.**
  That hides regressions. Detect environment capabilities separately
  (e.g. a module-level probe or fixture) and use `pytest.mark.skipif`.

## Types

- Name complex annotations with `type` aliases.
- `TypedDict` or `BaseModel` for fixed-key dicts. No `NamedTuple`.
- **Enums for finite sets.** When a variable accepts a known set of
  values (commands, provider names, panel variants, etc.), define a
  `StrEnum` or `IntEnum`. This lets the type checker catch typos and
  missing branches. Use `Literal` only when an enum would be
  impractical (e.g. values come from a third-party API).

## Control flow

- Prefer `match`/`case` over `if`/`elif` chains.
  Pattern matching is more readable, catches missing cases with
  type checkers, and destructures cleanly.

## Tool configuration

- Never change linter, formatter, or tool settings in `pyproject.toml`
  (ruff, rumdl, mypy, etc.) without explicit instructions.
- When lint rules flag issues, fix the code — don't loosen the rules.

## Suppressions

- Always specify exact code (`# noqa: S602`, `# type: ignore[assignment]`).
- Every `# type: ignore` **must** have a short reason on the same line
  explaining why the suppression is necessary
  (e.g. `# type: ignore[misc]  # pydantic __init__ reload`).
- Fix the root cause first.

## Class design

- Composition over inheritance. Functions over methods when in doubt.
- `Protocol` for interfaces. Subclassing only for strict is-a (e.g. exceptions).
- Start with duplication — extract only when the pattern is clear.

## Imports

- All imports at module level by default.
- Nested (in-function) imports are allowed **only** when there is a
  concrete reason: circular dependency, heavy module that should not
  load at import time, or conditional availability. Every nested
  import **must** have a comment on the same or preceding line
  explaining why it is deferred.

## Inline foreign-language code

- Never embed multi-line code in another language (bash, SQL, JS, etc.)
  as string literals inside Python. Write it to its own file where it
  gets proper syntax highlighting, linting, and is inspectable in an
  editor. Load at runtime via `importlib.resources` or a path constant.
- Exception: one-liners or very short commands (e.g. a single
  `subprocess.run(["bash", "-c", "echo hi"])` call) are fine inline.

## Python

- Target **Python 3.13+**. Use modern features directly.
- Run `just check` after **every** meaningful change — especially
  immediately after any behavioral change (refactoring, extraction,
  moving code). Do not batch multiple changes before checking.
- No hacks. Match existing style.
- When using JS/TS code as reference, understand the intent
  and rewrite idiomatically in Python. Never transliterate
  JS patterns — no callback pyramids, no `Promise`-shaped code,
  no `null`-checks where `Optional` and pattern matching work,
  no hand-rolled retry/event loops when stdlib or library
  equivalents exist.
